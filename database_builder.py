import os
import argparse
import time
import re
import logging
import sys
from typing import List, Dict, Any
import chromadb
from transformers import AutoTokenizer

# Configure terminal output (gray text) and file output
class GrayFormatter(logging.Formatter):
    def format(self, record):
        # ANSI escape code for gray text is \033[90m
        return f"\033[90m{super().format(record)}\033[0m"

logger = logging.getLogger('database_builder')
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler('rag_pipeline.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
file_handler.setFormatter(file_formatter)

# Terminal handler (gray)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = GrayFormatter('%(asctime)s - [%(levelname)s] - %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def has_exotic_characters(text: str) -> bool:
    """Check if the text contains non-printable ASCII or basic punctuation."""
    # Allow basic printable ASCII: space (32) to tilde (126), plus newlines and tabs
    for char in text:
        if not (32 <= ord(char) <= 126 or char in '\n\r\t'):
            return True
    return False

def strip_exotic_characters(text: str) -> str:
    """Remove non-printable ASCII characters."""
    return ''.join(char for char in text if 32 <= ord(char) <= 126 or char in '\n\r\t')

def clean_text(text: str) -> str:
    """Clean up excess whitespace and convert to lowercase."""
    text = re.sub(r'[ \t]+', ' ', text)  # remove excess inline whitespace
    # Keep newlines for page/paragraph splitting, but compress multiples slightly
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.lower()

def parse_and_chunk_document(filename: str, content: str, tokenizer, max_tokens=500, overlap=50) -> List[Dict[str, Any]]:
    """
    Splits the document into chunks and attaches metadata based on pages and paragraphs.
    Assumes <page ###> delineates pages and double newline delineates paragraphs.
    """
    chunks = []

    # Split into pages based on <page ###> tags (case-insensitive due to clean_text)
    page_splits = re.split(r'<page\s+(\d+)>', content)

    # If the document doesn't start with a <page> tag, the first element is the content before the first page tag
    # We will assume it's page 1 if not specified.
    current_page = "1"

    # process splits
    if len(page_splits) == 1:
        # No page tags found
        pages = [(current_page, page_splits[0])]
    else:
        pages = []
        if page_splits[0].strip():
            pages.append(("1", page_splits[0]))

        for i in range(1, len(page_splits), 2):
            if i+1 < len(page_splits):
                pages.append((page_splits[i], page_splits[i+1]))

    chunk_id_counter = 0

    for page_num, page_content in pages:
        # Split into paragraphs based on double newline
        paragraphs = re.split(r'\n\s*\n', page_content)

        for para_num, paragraph in enumerate(paragraphs, start=1):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Tokenize the paragraph
            tokens = tokenizer.encode(paragraph, add_special_tokens=False)

            # If paragraph fits in one chunk
            if len(tokens) <= max_tokens:
                chunk_text = tokenizer.decode(tokens)
                chunks.append({
                    "id": f"{filename}_p{page_num}_para{para_num}_{chunk_id_counter}",
                    "text": chunk_text,
                    "metadata": {
                        "source": filename,
                        "page": str(page_num),
                        "paragraph": str(para_num)
                    }
                })
                chunk_id_counter += 1
            else:
                # Need to split paragraph into multiple chunks with overlap
                start_idx = 0
                while start_idx < len(tokens):
                    end_idx = min(start_idx + max_tokens, len(tokens))
                    chunk_tokens = tokens[start_idx:end_idx]
                    chunk_text = tokenizer.decode(chunk_tokens)

                    chunks.append({
                        "id": f"{filename}_p{page_num}_para{para_num}_{chunk_id_counter}",
                        "text": chunk_text,
                        "metadata": {
                            "source": filename,
                            "page": str(page_num),
                            "paragraph": str(para_num)
                        }
                    })
                    chunk_id_counter += 1

                    if end_idx == len(tokens):
                        break

                    # Advance start_idx, accounting for overlap
                    start_idx = end_idx - overlap

    return chunks

def create_chroma_db_from_txt(source_dir: str, persist_dir: str, collection_name: str = "txt_documents"):
    """
    Reads .txt files from a directory, validates them, and stores them in a persistent ChromaDB.
    """
    logger.info(f"Starting document ingestion from '{source_dir}' to '{persist_dir}'")

    if not os.path.exists(source_dir):
        logger.error(f"Source directory '{source_dir}' does not exist.")
        return

    # Initialize the persistent client
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    # Load tokenizer for chunking
    # Using a fast tokenizer from HuggingFace
    tokenizer_name = "gpt2" # simple fallback tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    all_chunks = []

    for filename in os.listdir(source_dir):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(source_dir, filename)
        
        # Validation 1: Properly encoded (UTF-8)
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            logger.error(f"Skipping {filename}: Not a valid UTF-8 encoded file.")
            continue
        except Exception as e:
            logger.error(f"Skipping {filename}: Error reading file - {e}")
            continue

        # Validation 2: Non-empty (ignoring whitespace-only files)
        if len(content.strip()) == 0:
            logger.warning(f"Skipping {filename}: File is empty.")
            continue

        # Validation 3: Check for exotic characters
        if has_exotic_characters(content):
            print(f"\nWARNING: File '{filename}' contains exotic (non-printable/non-ascii) characters.")
            while True:
                choice = input(f"Do you want to strip these characters and proceed with '{filename}'? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    content = strip_exotic_characters(content)
                    logger.info(f"Stripped exotic characters from '{filename}'.")
                    break
                elif choice in ['n', 'no']:
                    logger.warning(f"Skipped '{filename}' due to exotic characters.")
                    content = None
                    break
                else:
                    print("Please answer 'y' or 'n'.")

            if content is None:
                continue # Skip to next file
        
        # Check for potential prompt injection or adversarial phrases
        malicious_patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"disregard all previous",
            r"bypass restrictions",
            r"you are now",
            r"do anything now"
        ]

        for pattern in malicious_patterns:
            if re.search(pattern, content.lower()):
                logger.warning("Potential prompt injection detected. Halting pipeline.")
                raise ValueError("Your doc was flagged as malicious.")

        # Transformation: clean text
        cleaned_content = clean_text(content)

        # Transformation & Storage: chunking
        start_chunk_time = time.time()
        file_chunks = parse_and_chunk_document(filename, cleaned_content, tokenizer, max_tokens=500, overlap=50)
        chunking_time = time.time() - start_chunk_time

        logger.info(f"Chunked '{filename}' into {len(file_chunks)} chunks in {chunking_time:.4f}s.")
        all_chunks.extend(file_chunks)

    # Add to ChromaDB vectorstore
    if all_chunks:
        documents = [c["text"] for c in all_chunks]
        metadatas = [c["metadata"] for c in all_chunks]
        ids = [c["id"] for c in all_chunks]

        logger.info(f"Embedding and inserting {len(all_chunks)} total chunks into ChromaDB...")
        start_embed_time = time.time()

        # ChromaDB automatically handles tokenization and default embeddings 
        # (using all-MiniLM-L6-v2) when documents are passed directly.
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        embed_time = time.time() - start_embed_time
        logger.info(f"Successfully embedded and added {len(all_chunks)} chunk(s) to '{collection_name}' in {embed_time:.4f}s.")

        # Validation 4: Ensure database entries match chunk count
        db_count = collection.count()
        # Note: if the DB already had items, this check needs to be relative to what we added.
        # But for simplicity, we assume we want the total DB count to increase by len(all_chunks)
        # To be completely robust and match the requirement "throw errors ... if the database entries don't match the chunk count"
        # We will retrieve the items we just inserted by ID to verify they exist.

        result = collection.get(ids=ids, include=[])
        inserted_count = len(result['ids'])

        if inserted_count != len(all_chunks):
            logger.error(f"CRITICAL ERROR: Database chunk mismatch! Expected {len(all_chunks)} inserted, found {inserted_count}.")
            raise ValueError("Database chunk count mismatch. Ingestion failed.")
        else:
            logger.info("Database consistency check passed: Chunk count matches inserted records.")

    else:
        logger.info("No valid documents met the criteria.")

    return collection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB from .txt files.")
    parser.add_argument("--source", type=str, required=True, help="Path to the directory containing .txt files.")
    parser.add_argument("--db", type=str, required=True, help="Path where the ChromaDB should be stored.")

    args = parser.parse_args()
    
    os.makedirs(args.source, exist_ok=True)
    
    try:
        create_chroma_db_from_txt(args.source, args.db)
    except Exception as e:
        logger.error(f"Ingestion process failed: {e}")
        sys.exit(1)