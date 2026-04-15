import time
import logging
import sys
import json
from typing import List, Dict, Any
import chromadb
from pydantic import BaseModel, Field, ValidationError

# Dummy constants for constraints
MIN_SIMILARITY_THRESHOLD = 0.5 # Arbitrary threshold for Phase 2 failure
MAX_PROMPT_TOKENS = 2000 # Size limit of smallest model (Phase 2 constraint)

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
# Mock Model Server APIs
# ==========================================
def simulate_generation_model(model_name: str, prompt: str) -> Dict[str, Any]:
    """Simulates a generation API call."""
    start_time = time.time()

    # Simulate network/processing latency (0.5 to 1.5 seconds)
    import random
    time.sleep(random.uniform(0.5, 1.5))

    # Generate mock responses based on the model
    if model_name == "Qwen":
        answer = f"According to the context, here is what I found. (Qwen's detailed answer to the prompt...)"
    elif model_name == "Llama":
        answer = f"The documents suggest that... (Llama's concise answer...)"
    else: # Mistral
        answer = f"Based on the provided relevance weighted chunks: (Mistral's structured answer...)"

    # Simulate hallucination risk (Failure Point 3)
    if random.random() < 0.1:
         answer += " Also, I hallucinated this extra irrelevant fact!"
         logger.warning(f"Simulated hallucination risk: {model_name} generated potentially unfaithful content.")

    latency = time.time() - start_time

    # Simple mock token counter
    tokens_used = len(prompt.split()) + len(answer.split())

    return {
        "model": model_name,
        "answer": answer,
        "latency_sec": latency,
        "tokens_used": tokens_used
    }

def simulate_arbitrator_phi(prompt: str, model_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simulates Phi acting as an arbitrator.
    Returns the updated responses list with added 'score' and 'reasoning'.
    """
    start_time = time.time()
    time.sleep(1.0) # simulate latency

    import random

    scored_responses = []
    for resp in model_responses:
        # Mock scoring logic
        if "hallucinated" in resp["answer"]:
            score = round(random.uniform(1.0, 2.5), 1)
            reasoning = "Penalized for unfaithfulness/hallucinated content."
        else:
            score = round(random.uniform(3.5, 5.0), 1)
            reasoning = "Answer addresses the prompt and aligns with context."

        scored_resp = dict(resp)
        scored_resp["arbitrator_score"] = score
        scored_resp["arbitrator_reasoning"] = reasoning
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
        resp = simulate_generation_model(model_name, prompt)
        model_responses.append(resp)
        logger.info(f"{model_name} generated response in {resp['latency_sec']:.2f}s (Tokens: {resp['tokens_used']})")

    logger.info("Invoking Arbitrator (Phi) to evaluate responses...")
    scored_responses = simulate_arbitrator_phi(prompt, model_responses)

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
def run_pipeline(raw_input: dict, db_path: str = "./my_chromadb") -> Dict[str, Any]:
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
        return {"status": "skipped_due_to_low_relevance"}

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
    if len(prompt_words) > MAX_PROMPT_TOKENS:
        truncated_count = len(prompt_words) - MAX_PROMPT_TOKENS
        logger.warning(f"Context overflow risk mitigated! Truncating prompt by {truncated_count} tokens to fit {MAX_PROMPT_TOKENS} limit.")
        prompt_words = prompt_words[:MAX_PROMPT_TOKENS]
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
