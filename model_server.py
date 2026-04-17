import time
import torch
import gc
import os
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Define global variables so the lifespan manager can load and delete them
model = None
tokenizer = None

MODEL_MAPPING = {
    "Qwen": "Qwen/Qwen2.5-3B-Instruct",
    "Llama": "meta-llama/Llama-3.2-3B-Instruct",
    "Mistral": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    "Phi": "nvidia/Phi-4-reasoning-plus-NVFP4"
}

# 2. Create the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    global model, tokenizer

    model_key = os.environ.get("MODEL_NAME", "Qwen")
    hf_model_id = MODEL_MAPPING.get(model_key)

    if not hf_model_id:
        raise ValueError(f"Invalid MODEL_NAME environment variable: {model_key}. Must be one of {list(MODEL_MAPPING.keys())}")

    print(f"Loading {hf_model_id} (Alias: {model_key})...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print(f"Model {model_key} loaded!")
    
    yield  # The server runs while yielded here
    
    # --- SHUTDOWN LOGIC ---
    print("\nInitiating graceful shutdown...")
    
    # Remove references to the massive objects
    del model
    del tokenizer
    
    # Force Python to garbage collect the unreferenced objects
    gc.collect()
    
    # Force PyTorch to release the memory back to the OS/GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("CUDA memory freed!")
    
    print("Cleanup complete. Goodbye!")

# 3. Pass the lifespan to FastAPI
app = FastAPI(title="Local OpenAI-Compatible Server", lifespan=lifespan)

# --- OpenAI-compatible schemas ---
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    stream: Optional[bool] = False

# --- Chat Endpoint ---
@app.post("/v1/chat/completions")
async def generate_chat(request: ChatCompletionRequest):
    # (Your existing generation code remains exactly the same here)
    try:
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        inputs = tokenizer.apply_chat_template(
            conversation=messages_dict,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=request.max_tokens,
                temperature=request.temperature if request.temperature > 0 else 0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        input_length = inputs["input_ids"].shape[1] 
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                },
                "finish_reason": "stop"
            }]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Programmatic Shutdown Endpoint (Optional) ---
@app.post("/shutdown")
async def shutdown(background_tasks: BackgroundTasks):
    """
    Allows an external script to gracefully kill the server.
    We use a background task to send the signal so the API can return a 200 OK first.
    """
    def kill_server():
        time.sleep(1) # Give the server a second to return the response
        os.kill(os.getpid(), signal.SIGINT) # Triggers the lifespan shutdown logic
        
    background_tasks.add_task(kill_server)
    return {"message": "Shutdown signal received. Freeing memory and terminating..."}