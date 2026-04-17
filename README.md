# Multi-Agent RAG Pipeline

This project implements a multi-agent Retrieval-Augmented Generation (RAG) pipeline. It uses multiple language models (Qwen, Llama, and Mistral) to generate answers simultaneously, and a separate arbitrator model (Phi) to evaluate and select the best response.

## Architecture

To manage VRAM efficiently and allow independent scaling, this system uses a **microservice architecture**. Each language model is hosted on its own dedicated Uvicorn server running on a specific local port.

The `pipeline.py` script routes requests to the correct model server based on the following port mapping:
- **Qwen** (`Qwen/Qwen2.5-3B-Instruct`): Port 8000
- **Llama** (`meta-llama/Llama-3.2-3B-Instruct`): Port 8001
- **Mistral** (`mistralai/Ministral-3-3B-Instruct-2512-BF16`): Port 8002
- **Phi** (`nvidia/Phi-4-reasoning-plus-NVFP4`): Port 8003

## How to Run

### Step 1: Start the Model Servers
You must start a separate server instance for each model you want the pipeline to use. Open four separate terminal windows and run the following commands:

**Terminal 1 (Qwen):**
```bash
MODEL_NAME="Qwen" uvicorn model_server:app --port 8000
```

**Terminal 2 (Llama):**
```bash
MODEL_NAME="Llama" uvicorn model_server:app --port 8001
```

**Terminal 3 (Mistral):**
```bash
MODEL_NAME="Mistral" uvicorn model_server:app --port 8002
```

**Terminal 4 (Phi - Arbitrator):**
```bash
MODEL_NAME="Phi" uvicorn model_server:app --port 8003
```

Wait until all servers indicate they have successfully loaded their models and are accepting connections.

### Step 2: Run the Pipeline
Once all the necessary model servers are running, open a new terminal window and run the main pipeline program. You may optionally provide a path to your ChromaDB instance as an argument.

```bash
python pipeline.py ./my_chroma_db
```

The program will prompt you to enter questions and will coordinate with the local microservices to generate an arbitrated answer.
