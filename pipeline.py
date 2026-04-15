import time
import logging
from typing import List
import chromadb
from pydantic import BaseModel, Field, ValidationError

from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure intermediate logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

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
        print(f"Answer: {self.answer}\n")
        print(f"Sources: {', '.join(self.sources)}\n")
        print(f"Processing Time: {self.processing_time_ms} ms")

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_collection(name="txt_documents")

# Connect to the local model at 127.0.0.1:8000
llm = OpenAI(
    base_url="http://127.0.0.1:8000/v1", 
    api_key="local",
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Answer the question strictly using the provided context:\n\n{context}"),
    ("human", "{question}")
])

# ==========================================
# Main RAG Pipeline
# ==========================================
def run_pipeline(raw_input: dict) -> dict:
    start_time = time.time()
    
    # Step A: Accept Input & Validate Schema
    try:
        validated_input = QueryInput(**raw_input)
        logger.info(f"Schema validation passed. Query: '{validated_input.question}'")
    except ValidationError as e:
        logger.error(f"Input validation failed: {e}")
        raise ValueError(f"Invalid input schema: {e}")

    # Step B: Transform Representation (Retrieval)
    logger.info("Transforming query into vector representation and retrieving context...")
    results = collection.query(
        query_texts=[validated_input.question],
        n_results=2,
        include=["documents", "metadatas"]
    )
    retrieved_docs = results["documents"][0] if results["documents"] else []
    
    context_text = "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant context found."
    sources = [doc["source"] for doc in results["metadatas"][0]] if results["metadatas"] else []
    logger.info(f"Retrieval complete. Found {len(retrieved_docs)} document(s). Sources: {sources}")

    # Step C: Call the Model
    logger.info("Invoking the local language model...")
    prompt = prompt_template.format(context=context_text, question=validated_input.question)
    response = llm.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
        stream=False)
    raw_response = response.choices[0].message.content
    logger.info("Language model generation complete.")

    # Step D: Postprocess Output
    logger.info("Postprocessing and validating output schema...")
    processing_time = (time.time() - start_time) * 1000
    
    final_output = RAGOutput(
        answer=raw_response.strip(),
        sources=sources,
        processing_time_ms=round(processing_time, 2)
    )
    
    logger.info(f"Pipeline finished successfully in {final_output.processing_time_ms} ms.")
    
    return final_output

# ==========================================
# 4. Execution Example
# ==========================================
if __name__ == "__main__":
    while True:
        user_input = input("Enter your question: ")

        # produce a string of arbitrary length to test the pipeline's handling of input size
        # for _ in range(100000):
        #     user_input += " This is additional context to increase the input size and test the pipeline's robustness." 
        test_input = {"question": user_input}
        print(f"Testing pipeline with input of length {len(test_input['question'])} characters...")
        
        print("\n--- Starting Local Pipeline Execution ---")
        try:
            result = run_pipeline(test_input)
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            exit(1)
        
        print("\n--- Final Output ---")
        result.print()