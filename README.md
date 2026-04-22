# Multi-Agent RAG Pipeline

This project implements a multi-agent Retrieval-Augmented Generation (RAG) pipeline. It uses multiple language models (Qwen, Llama, and Mistral) to generate answers simultaneously, and a separate arbitrator model (Phi) to evaluate and select the best response.

## Architecture

To manage VRAM efficiently and allow independent scaling, this system uses a **microservice architecture**. Each language model is hosted on its own dedicated Uvicorn server running on a specific local port.

The `pipeline.py` script routes requests to the correct model server based on the following port mapping:
- **Qwen** (`Qwen/Qwen2.5-3B-Instruct`): Port 8000
- **Llama** (`meta-llama/Llama-3.2-3B-Instruct`): Port 8001
- **Mistral** (`mistralai/Mistral-7B-Instruct-v0.2`): Port 8002
- **Phi** (`microsoft/phi-4`): Port 8003

## Document Ingestion

Before running the pipeline, you must ingest the documents to build the ChromaDB database.
To ingest the documents from the `rome` directory and store them in `my_chroma_db`, run the following command:

```bash
python database_builder.py --source rome --db my_chroma_db
```

*Note: Document ingestion must happen prior to pipeline runtime.*

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

You have two options for running the pipeline: a Command Line Interface (CLI) or a Gradio Web Interface.

#### Option A: Gradio Web Interface (Recommended)
To run the interactive web interface, open a new terminal window and run:

```bash
python ui_pipeline.py
```

This will start a local Gradio server bound to `0.0.0.0`. Open the provided URL (usually `http://127.0.0.1:7860` locally) in your web browser to interact with the RAG pipeline using a chat interface.

##### Accessing over SSH
If you are running the Gradio interface on a remote machine, you can access it securely using SSH port forwarding.
1. On your local computer, open a terminal and run the following command to create an SSH tunnel:
   ```bash
   ssh -L 7860:localhost:7860 user@<remote-host-ip>
   ```
2. Once connected, open your local web browser and go to `http://127.0.0.1:7860`.

##### Accessing over the Local Network
If you are on the same network as the host machine, you can connect directly via its IP address:
1. Find the host's IP address:
   - **Linux/macOS:** Run `hostname -I` or `ifconfig`.
   - **Windows:** Run `ipconfig` and look for the IPv4 Address.
2. In your web browser, navigate to `http://<host-ip>:7860`.
*(Note: You may need to configure the host machine's firewall to allow traffic on port 7860).*

#### Option B: Command Line Interface (CLI)
Once all the necessary model servers are running, open a new terminal window and run the main pipeline program. You may optionally provide a path to your ChromaDB instance as an argument.

```bash
python pipeline.py ./my_chroma_db
```

The program will prompt you to enter questions and will coordinate with the local microservices to generate an arbitrated answer.
