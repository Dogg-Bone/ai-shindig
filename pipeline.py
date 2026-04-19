import time
import logging
import sys
import json
import re
from typing import List, Dict, Any
import chromadb
from pydantic import BaseModel, Field, ValidationError

# Dummy constants for constraints
MIN_SIMILARITY_THRESHOLD = 0.5 # Arbitrary threshold for Phase 2 failure
MAX_PROMPT_WORDS = 2000 # Size limit of smallest model (Phase 2 constraint)

# Configure terminal output (gray text) and file output
class GrayFormatter(logging.Formatter):
    def format(self, record):
        return f"\033[90m{super().format(record)}\033[0m"

logger = logging.getLogger('rag_pipeline')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('rag_pipeline.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = GrayFormatter('%(asctime)s - [%(levelname)s] - %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ==========================================
# Schema Validation Definitions
# ==========================================
class QueryInput(BaseModel):
    question: str = Field(..., min_length=3, max_length=900, description="The user query to be answered.")

class RAGOutput(BaseModel):
    answer: str
    sources: List[str]
    processing_time_ms: float
    def print(self):
        print(f"\nFinal Answer: {self.answer}\n")
        print(f"Sources: {', '.join(self.sources)}\n")
        print(f"Total Processing Time: {self.processing_time_ms} ms")

# ==========================================
# Real Model Server APIs
# ==========================================
import requests
from tenacity import retry, stop_after_attempt

# Define ports for each microservice
PORT_MAPPING = {
    "Qwen": 8000,
    "Llama": 8001,
    "Mistral": 8002,
    "Phi": 8003
}

@retry(stop=stop_after_attempt(3))
def _call_generation_api(model_name: str, prompt: str) -> Dict[str, Any]:
    # Look up the port for the specified model
    port = PORT_MAPPING.get(model_name, 8000)
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}]
    }
    # Assumption: The server returns a standard OpenAI JSON response:
    # {"choices": [{"message": {"content": "..."}}], "usage": {"total_tokens": 123}}
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()

def call_generation_model(model_name: str, prompt: str) -> Dict[str, Any]:
    """Calls a generation API."""
    start_time = time.time()

    try:
        api_response = _call_generation_api(model_name, prompt)
        answer = api_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens_used = api_response.get("usage", {}).get("total_tokens", 0)
    except Exception as e:
        logger.error(f"Failed to call {model_name} after 3 attempts: {e}")
        answer = f"Error generating response: {e}"
        tokens_used = 0

    latency = time.time() - start_time

    return {
        "model": model_name,
        "answer": answer,
        "latency_sec": latency,
        "tokens_used": tokens_used
    }

@retry(stop=stop_after_attempt(3))
def _call_arbitrator_api(prompt: str, model_answer: str) -> Dict[str, Any]:
    # Assumption: The local server provides an OpenAI-compatible endpoint.
    port = PORT_MAPPING.get("Phi", 8003)
    url = f"http://localhost:{port}/v1/chat/completions"

    arbitrator_prompt = f"""You are a strict, impartial judge evaluating a model's response.
Evaluate the answer based strictly on the following two criteria, in order of importance:
1. Faithfulness: Is the answer derived *only* from the System Context provided in the Prompt? Any hallucinations or inclusion of outside knowledge should be severely penalized.
2. Relevance: Does the answer directly and accurately address the User Question?

Provide a score from 1.0 to 5.0. You must output the result in exactly the following format:
Score: <your float score here>
Reasoning: <your reasoning here>

Prompt (includes Context and Question):
{prompt}

Answer to Evaluate:
{model_answer}"""

    payload = {
        "model": "Phi",
        "messages": [{"role": "user", "content": arbitrator_prompt}]
    }
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()

def call_arbitrator_phi(prompt: str, model_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calls Phi acting as an arbitrator.
    Returns the updated responses list with added 'score' and 'reasoning'.
    """
    start_time = time.time()
    scored_responses = []

    for resp in model_responses:
        scored_resp = dict(resp)
        try:
            api_response = _call_arbitrator_api(prompt, resp["answer"])
            content = api_response.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Assumption: The model will respond with a JSON-like format or plain text that we can extract score from.
            # For simplicity, we assume a structured text like "Score: 4.5\nReasoning: It is good."
            # In a real scenario, structured output/JSON mode would be preferred.
            # Here we parse naively based on assumptions.
            score = 3.0 # Default score
            reasoning = content

            if "Score:" in content:
                try:
                    score_str = content.split("Score:")[1].split()[0].strip()
                    # Remove markdown asterisks or other non-numeric formatting
                    score_str = re.sub(r'[^\d.]', '', score_str)
                    score = float(score_str)
                except Exception as e:
                    logger.warning(f"Could not parse score from Phi response: {e}")

            scored_resp["arbitrator_score"] = score
            scored_resp["arbitrator_reasoning"] = reasoning

        except Exception as e:
            logger.error(f"Failed to call Phi arbitrator after 3 attempts: {e}")
            scored_resp["arbitrator_score"] = 0.0
            scored_resp["arbitrator_reasoning"] = f"Error evaluating response: {e}"

        scored_responses.append(scored_resp)

    latency = time.time() - start_time
    logger.info(f"Arbitrator (Phi) evaluation completed in {latency:.2f}s.")
    return scored_responses

# ==========================================
# Phase 2 Helper Functions
# ==========================================
from database_builder import clean_text, has_exotic_characters, strip_exotic_characters

def prepare_and_validate_query(raw_query: str) -> str:
    """Applies the same cleaning and validation to the query as the ingestion phase."""
    if has_exotic_characters(raw_query):
        logger.warning("Query contains exotic characters. Attempting to strip.")
        raw_query = strip_exotic_characters(raw_query)

    cleaned_query = clean_text(raw_query)
    return cleaned_query

# ==========================================
# Phase 3 & 4 Main Execution Logic
# ==========================================
def execute_models_and_arbitrate(prompt: str, metadatas: List[Dict[str, Any]], similarity_scores: List[float], start_total_time: float) -> Dict[str, Any]:
    """Handles generating model responses and invoking the arbitrator."""
    # Phase 3: Model Inference (The Multi-Agent Core)
    logger.info("Sending prompt simultaneously to generating models (Qwen, Llama, Mistral)...")

    # Simulating sequential execution of the "simultaneous" models
    model_responses = []
    for model_name in ["Qwen", "Llama", "Mistral"]:
        resp = call_generation_model(model_name, prompt)
        model_responses.append(resp)
        logger.info(f"{model_name} generated response in {resp['latency_sec']:.2f}s (Tokens: {resp['tokens_used']})")

    logger.info("Invoking Arbitrator (Phi) to evaluate responses...")
    scored_responses = call_arbitrator_phi(prompt, model_responses)

    for sr in scored_responses:
        logger.info(f"Arbitrator Score for {sr['model']}: {sr['arbitrator_score']} - Reason: {sr['arbitrator_reasoning']}")

    # Phase 4: Postprocessing and Output
    # Select the highest scoring response
    winning_response = max(scored_responses, key=lambda x: x['arbitrator_score'])
    logger.info(f"Model '{winning_response['model']}' won with a score of {winning_response['arbitrator_score']}.")

    # Format precise citations
    sources_formatted = []
    for meta, score in zip(metadatas, similarity_scores):
        sources_formatted.append(f"[{meta['source']}, Page {meta['page']}, Paragraph {meta['paragraph']}] (Relevance: {score:.2f})")

    processing_time = (time.time() - start_total_time) * 1000

    logger.info("Pipeline execution finished.")

    final_output = RAGOutput(
        answer=winning_response["answer"],
        sources=sources_formatted,
        processing_time_ms=round(processing_time, 2)
    )

    return {"status": "success", "output": final_output}

# ==========================================
# Main RAG Pipeline
# ==========================================
def run_pipeline(raw_input: dict, db_path: str = "./my_chroma_db") -> Dict[str, Any]:
    start_total_time = time.time()
    
    # Connect to database
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name="txt_documents")
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        raise

    # Step A: Accept Input & Validate Schema
    try:
        validated_input = QueryInput(**raw_input)
        logger.info(f"Schema validation passed. Query: '{validated_input.question}'")
    except ValidationError as e:
        logger.error(f"Input validation failed: {e}")
        raise ValueError(f"Invalid input schema: {e}")

    # Phase 2: Intake and Prompt Preparation
    cleaned_query = prepare_and_validate_query(validated_input.question)

    logger.info("Transforming query into vector representation and retrieving context...")
    retrieval_start = time.time()

    # We retrieve n_results and their distances (which map to relevance scores)
    # ChromaDB returns distances. Lower distance = higher similarity.
    # We convert distance to a mock similarity score (1.0 - distance/max_expected_distance)
    # For simplicity, we just invert the distance if it's L2, or treat it directly if cosine.
    results = collection.query(
        query_texts=[cleaned_query],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    retrieval_latency = time.time() - retrieval_start
    logger.info(f"Retrieval completed in {retrieval_latency:.4f}s.")

    retrieved_docs = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    # Calculate mock similarity scores from distances (assuming cosine distance 0..2)
    similarity_scores = [max(0.0, 1.0 - (d / 2.0)) for d in distances]
    
    # Log scores
    logger.info(f"Retrieved {len(retrieved_docs)} documents. Similarity scores: {[round(s,2) for s in similarity_scores]}")

    # Policy Enforcement: Similarity Threshold
    if not similarity_scores or max(similarity_scores) < MIN_SIMILARITY_THRESHOLD:
        logger.warning(f"Similarity threshold failure (max score {max(similarity_scores, default=0.0):.2f} < {MIN_SIMILARITY_THRESHOLD}). Skipping AI models.")
        print("\n" + "="*50)
        print("POLICY ENFORCEMENT: LOW RELEVANCE")
        print("The system did not find sufficiently relevant information to answer your question automatically.")
        print("Here are the top retrieved documents for your manual review:")
        print("="*50)
        for i, doc in enumerate(retrieved_docs):
            meta = metadatas[i]
            print(f"\n--- Document {i+1} ---")
            print(f"Citation: [{meta['source']}, Page {meta['page']}, Paragraph {meta['paragraph']}]")
            print(f"Relevance Score: {similarity_scores[i]:.2f}")
            print(f"Content: {doc}")
        print("="*50)
        # Exit gracefully
        return {
            "status": "skipped_due_to_low_relevance",
            "docs": retrieved_docs,
            "metadatas": metadatas,
            "scores": similarity_scores
        }

    # Assemble Prompt
    # Append mathematical weightings to each chunk
    context_chunks_formatted = []
    for doc, meta, score in zip(retrieved_docs, metadatas, similarity_scores):
        context_chunks_formatted.append(f"[Relevance Score: {score:.2f}]\n{doc}")

    context_text = "\n\n".join(context_chunks_formatted)
    
    # Separate question and context clearly
    raw_prompt = f"System Context:\n{context_text}\n\nUser Question:\n{cleaned_query}"
    
    # Enforce strict size limits of the smallest AI model (MAX_PROMPT_TOKENS)
    # We use a simple word split for mock tokenization
    prompt_words = raw_prompt.split()
    if len(prompt_words) > MAX_PROMPT_WORDS:
        truncated_count = len(prompt_words) - MAX_PROMPT_WORDS
        logger.warning(f"Context overflow risk mitigated! Truncating prompt by {truncated_count} tokens to fit {MAX_PROMPT_WORDS} limit.")
        prompt_words = prompt_words[:MAX_PROMPT_WORDS]
        prompt = " ".join(prompt_words)
    else:
        prompt = raw_prompt

    logger.info("Phase 2 complete. Prompt prepared successfully.")

    return execute_models_and_arbitrate(prompt, metadatas, similarity_scores, start_total_time)

# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    db_path = "./my_chroma_db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        
    while True:
        try:
            user_input = input("\nEnter your question (or type 'quit' to exit): ")
            if user_input.lower() in ['quit', 'exit']:
                break

            test_input = {"question": user_input}

            print("\n--- Starting Multi-Agent RAG Pipeline ---")
            result = run_pipeline(test_input, db_path=db_path)

            if result["status"] == "success":
                result["output"].print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            break
