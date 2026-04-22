# Section 4: Architecture Document
## 4.1 System Overview
This multi-model RAG pipeline solves the problem of answering complex questions by maximizing response quality and faithfulness. Instead of breaking down complex web searches, the new system generates multiple answers from different LLMs and arbitrates them against local documents to find the best response. The primary users are researchers or general users requiring synthesized findings from specific documents, rigorously evaluated for faithfulness and relevance. Figure 1 shows the high-level architecture of the system.

```mermaid
graph TD;
    USER_QUERY([USER QUERY])
    CHROMA[(CHROMA DATABASE)]
    CONTEXT_PACKAGING[CONTEXT PACKAGING]
    GEN_A[GENERATOR A (QWEN)]
    GEN_B[GENERATOR B (LLAMA)]
    GEN_C[GENERATOR C (MISTRAL)]
    JUDGE[JUDGE (PHI)]
    RESPONSE([RESPONSE])

    USER_QUERY -->|DIRECT| CHROMA
    USER_QUERY -->|DIRECT| CONTEXT_PACKAGING
    CHROMA -->|DIRECT| CONTEXT_PACKAGING

    CONTEXT_PACKAGING -->|HTTP| GEN_A
    CONTEXT_PACKAGING -->|HTTP| GEN_B
    CONTEXT_PACKAGING -->|HTTP| GEN_C

    GEN_A -->|HTTP| JUDGE
    GEN_B -->|HTTP| JUDGE
    GEN_C -->|HTTP| JUDGE

    CHROMA -->|DIRECT| JUDGE
    
    JUDGE -->|DIRECT| RESPONSE

    classDef blue fill:#DAE8FC,stroke:#6C8EBF
    classDef green fill:#D5E8D4,stroke:#82B366
    classDef gray fill:#F5F5F5,stroke:#666666
    classDef orange fill:#FFE6CC,stroke:#D79B00
    classDef yellow fill:#FFF2CC,stroke:#D6B656
    classDef red fill:#F8CECC,stroke:#B85450
    classDef purple fill:#E1D5E7,stroke:#9673A6

    class USER_QUERY,RESPONSE blue
    class CHROMA green
    class CONTEXT_PACKAGING gray
    class GEN_A orange
    class GEN_B yellow
    class GEN_C red
    class JUDGE purple
```
**Figure 1** System architecture diagram

The system accepts a natural language query, validates the schema using Pydantic, retrieves relevant context from a local vector database, generates answers concurrently using three distinct models (Qwen, Llama, Mistral), and finally arbitrates the responses using a Phi model to select the highest quality output.

### Boundaries
**Pipeline to Local LLMs (HTTP)**: Potential risk of server overload, latency spikes, or timeouts when calling models on ports `8000`, `8001`, `8002`, and `8003`. *Failure mode:* Retries local inference requests up to 3 times using the `tenacity` library before handling the failure gracefully.

**Pipeline to ChromaDB (Local Direct)**: Corrupted ChromaDB indices or missing persistent directories. *Failure mode:* The system prevents hallucination by requiring retrieved context. If no relevant context is retrieved, the similarity threshold policy trips, and the AI execution is bypassed completely.

## 4.2 Pipeline Phases
### 4.2.1 Phase 1 - Schema Validation
1. **Role**: Ensures that user input meets strict size and formatting requirements before any processing occurs. Uses Pydantic (`QueryInput`).
2. **Inputs:** Raw user string input.
3. **Outputs:** Validated `QueryInput` object.
4. **Tools:** Pydantic validation functions.
5. **Context Management:** Enforces a character limit of 3 to 900 characters for the user query to prevent abuse and manage context size upfront.
6. **Confidence Signaling:** Explicit error throwing. If validation fails, an exception is raised and execution halts immediately.
7. **Handoff Schema:** Passes the validated query string downstream.

### 4.2.2 Phase 2 - Intake & Prompt Preparation
1. **Role**: Sanitizes the query, retrieves documents from ChromaDB, converts distance to similarity, enforces relevance policies, and constructs the final prompt within token limits.
2. **Inputs:** Validated query string.
3. **Outputs:** System prompt (context + query), retrieved metadata, and similarity scores.
4. **Tools:** `clean_text`, `strip_exotic_characters`, ChromaDB client.
5. **Context Management:** Limits retrieved context to the top 3 documents. Calculates token limits via word count estimation, truncating the total prompt to `MAX_PROMPT_WORDS` (2000) to ensure the smallest model can process it without overflowing.
6. **Confidence Signaling:** Compares highest retrieved similarity score against `MIN_SIMILARITY_THRESHOLD` (0.5). If no document meets the threshold, execution halts and returns a low-confidence notice to the user, acting as a circuit breaker against hallucination.
7. **Handoff Schema:** Passes the formatted prompt string, source metadata, and similarity scores to the inference models.

### 4.2.3 Phase 3 - Model Inference
1. **Role**: Dispatches the identical prompt to three parallel generation models to produce diverse answers.
2. **Inputs:** Formatted prompt string.
3. **Outputs:** List of response dictionaries.
4. **Tools:** HTTP POST requests to local LLM server ports (Qwen: 8000, Llama: 8001, Mistral: 8002).
5. **Context Management:** Relies entirely on the prompt formatting enforced in Phase 2. No dynamic compression occurs during inference.
6. **Confidence Signaling:** Tracks individual model latency and token usage. Uses the `tenacity` library to retry failed API calls up to 3 times before returning an error string for that specific model.
7. **Handoff Schema:** Aggregates the responses into an array of dictionaries containing the model name, generated text, latency, and tokens used.

### 4.2.4 Phase 4 - Arbitration
1. **Role**: Evaluates the generated answers against strict faithfulness and relevance criteria to determine the best response.
2. **Inputs:** Formatted prompt string, list of response dictionaries from Phase 3.
3. **Outputs:** Final response output including the highest-scoring answer and exact source citations.
4. **Tools:** HTTP POST request to the local Phi model port (8003).
5. **Context Management:** Feeds both the original context and the generated answer into the Phi model's context window.
6. **Confidence Signaling:** The Phi model outputs a numerical float score (1.0 to 5.0) and reasoning. If parsing the float fails, the system safely defaults the score to `3.0`.
7. **Handoff Schema:** Extracts the winning answer based on the highest score and formats the final `RAGOutput` object with citations and processing time for the user.
## 4.3 Retrieval Architecture
- **Chunking Strategy:** Recursive character text splitting with a maximum chunk size of 1000 characters and a 200 character overlap. This was chosen to maintain paragraphs intact where possible, while also respecting `all-MiniLM-L6-v2`'s limits. The optimal token count for this embedding model is 256, but the maximum is 512 tokens.
- **Embedding Model:** `HuggingFaceEmbeddings` using `all-MiniLM-L6-v2`. Chosen for being lightweight, not too slow for internal inference, specializing in use for embeddings, and the capacity to run it locally at no API cost. Should a superior model become available, the swap is as simple as changing the `model_name` parameter in initialization, wiping the ChromaDB vector store, and re-ingesting project documents (there will be a limited amount, so this will not be prohibitively expensive).
- **Retrieval Evaluation:** The retrieval step returns a maximum of three docs per search. Therefore retrieval can be evaluated using the metrics *Precision@3* and *nDCG@3*. *Precision@3* measures how many of the top 3 retrieved docs belong to the expected source docs. *nDCG@3* is similar, but also evaluates the order of those retrieved docs. Both metrics require a unit test dataset, so will not be able to track performance during deployment.
- **Distance to Similarity Conversion:** ChromaDB inherently returns L2 distance metrics (lower is better). In `pipeline.py`, this is mathematically converted into a mock similarity score ranging from 0.0 to 1.0 using the formula `max(0.0, 1.0 - (distance / 2.0))`. This provides an intuitive percentage-based score that is used for threshold policy enforcement.
- **Sanitization Shift:** Web-based HTML and Markdown stripping has been removed. Instead, the database builder relies on a `has_exotic_characters` check and `strip_exotic_characters` logic to clean local text files before ingestion, ensuring clean tokenization for the embedding model.
## 4.4 Reliability and Security Decisions
- **API Retries:** The `tenacity` library handles potential local network and API failures when communicating with the microservice LLM ports (`8000`-`8003`). The system makes up to 3 attempts before catching the exception and returning an error string, preventing temporary port conflicts from halting the pipeline.
- **Failsafe Parsing:** The Arbitrator model (Phi) is prompted to return a specific "Score: X.X" format. If the model fails to adhere to this format and raises a parsing exception, the system catches it and defaults to a safe score of `3.0`. This prevents strict parsing errors from crashing the final response selection.
- **Policy Enforcement (Circuit Breaker):** The pipeline enforces a rigid similarity cutoff (`MIN_SIMILARITY_THRESHOLD` of `0.5`). If ChromaDB fails to find documents relevant enough to exceed this score, the pipeline bypasses LLM generation entirely and informs the user. This is the primary defense against hallucination.
- **Context Overflow Mitigation:** The pipeline protects the inference models from out-of-memory errors or context window overflows by aggressively truncating the final constructed prompt. If the prompt exceeds `MAX_PROMPT_WORDS` (2000), it is sliced to fit the limits of the smallest generation model in the array.
- **Input Schema Validation:** The very first step of the pipeline uses Pydantic to enforce string boundaries (between 3 and 900 characters) on the user query. This protects against buffer overflows, abnormally massive prompts, and empty submissions.
## 4.5 Deployment Plan
- **Container strategy:** For production, this application requires an orchestrated container approach, such as `docker-compose`. The setup must manage the central pipeline container alongside four distinct LLM endpoint containers (`model_server.py`) mapped to ports `8000`-`8003`.
- **Secrets management:** As the system relies entirely on local models and a local ChromaDB, it currently requires no external API keys, reducing the need for complex secrets management.
- **State management:** State management via LangGraph checkpointers has been removed entirely. The system is stateless across requests, relying only on the persistent ChromaDB directory (`./my_chroma_db`) for knowledge retention.
- **Edge considerations:** The multi-model approach makes edge deployment difficult. While the embedding model and a single LLM (like Qwen) could run on a CPU, simultaneously hosting Qwen, Llama, Mistral, and Phi requires significant RAM and compute, favoring a centralized, GPU-backed server deployment.

# Section 6: Failure Injection Report
**Overview:**
- **Failure 1:** Arbitrator Parsing Failure
- **Failure 2:** ChromaDB Low Relevance Threshold

---
#### Failure Injection 1: Arbitrator Parsing Failure
**1. Code/Configuration Change**: To simulate the Phi model outputting an unexpected format, modify the `call_arbitrator_phi` function in `pipeline.py`. Around line 105, directly inject a bad string:
```python
            # ... existing code ...
            api_response = _call_arbitrator_api(prompt, resp["answer"])
            # INJECT BAD CONTENT HERE
            content = "This is a great response! I give it five stars."
            # content = api_response.get("choices", [{}])[0].get("message", {}).get("content", "")
```
Run the pipeline and ask a valid question.

**2. Observed System Behavior**:
```text
[Paste your logs here]
```

**3. Failure Nature**: The arbitrator fails to output a strict `Score: X.X` float, which ordinarily would cause a float conversion error and crash the pipeline right before the final response.
**4. Propagation Boundary**: The error is caught within the `call_arbitrator_phi` function's `try/except` block (or the parsing fallback).
**5. Architectural Control Mitigation**: The system is designed with Failsafe Parsing. It catches the parsing error and defaults the `arbitrator_score` to `3.0`. The pipeline successfully completes and outputs a response, rather than crashing.

---
#### Failure Injection 2: ChromaDB Low Relevance Threshold
**1. Code/Configuration Change**: To simulate a failure to find relevant documents, ask a question completely unrelated to the database contents, or forcefully adjust the threshold logic. In `pipeline.py`, around line 186, temporarily raise the threshold:
```python
    # Change threshold to 0.99 to force failure
    if not similarity_scores or max(similarity_scores) < 0.99: # MIN_SIMILARITY_THRESHOLD
```
Run the pipeline and ask any question.

**2. Observed System Behavior**:
```text
[Paste your logs here]
```

**3. Failure Nature**: The system fails to retrieve any documents that meet the minimum relevance criteria required to formulate a faithful prompt.
**4. Propagation Boundary**: The pipeline execution halts at Phase 2 (Intake & Prompt Preparation). The generation and arbitrator models are never called.
**5. Architectural Control Mitigation**: The Policy Enforcement circuit breaker trips. Instead of allowing the LLMs to guess or hallucinate without context, the system cleanly aborts, displaying a policy enforcement message and dumping the highest-scoring (but rejected) documents for the user to review manually.

# Section 7: Reflection
### Question 1: The PA2 Comparison
In PA3, the agentic paradigm relied heavily on natural language instructions for routing and structuring output (e.g., asking the Decomposer to output a JSON list), which was fragile and prone to silent failures if the model decided to chat instead of format. This multi-model RAG pipeline reverts to the strictness of PA2, using Pydantic schema enforcement at the input layer and python-native parsing fallbacks at the arbitrator layer. This trades the "smart" flexibility of an agent for the deterministic reliability of a pipeline.

### Question 2: Your Most Fragile Component
The most fragile component in this new pipeline is the Arbitrator (Phi) evaluation. While Pydantic protects the input, the final response selection relies entirely on the Phi model correctly parsing its instructions and outputting a consistent score format ("Score: X.X"). While the failsafe default of `3.0` prevents a crash, if the model consistently hallucinates its formatting, the pipeline degrades into randomly selecting one of the generated answers without actual quality arbitration.

### Question 3: The Model Replacement Test
If the local LLMs (Qwen, Llama, Mistral) were replaced with larger cloud models (like GPT-4), the pipeline's structure would remain identical, but the generation phase would likely take longer due to network latency, while the quality of answers would improve. However, if the Arbitrator model (Phi) was swapped for a model with different alignment, it might weigh the "Faithfulness" vs "Relevance" criteria differently, radically altering which generator "wins" the arbitration phase. The rigid `MIN_SIMILARITY_THRESHOLD` might also need tuning if a new embedding model produces different distance distributions.

### Question 4: What You Would Build Differently
Trading the agentic system for this pipeline drastically increased reliability but removed the system's ability to iteratively research a topic on the internet. If I were to build this again, I would attempt a hybrid approach. I would keep the strict multi-model generation and arbitration (to ensure maximum answer quality), but I would wrap it inside a single Manager agent. If the pipeline's Arbitrator determines that *all* generated responses scored poorly (e.g., all scores < 2.0), it could return control to the Manager, which could then execute a web-search tool to gather more context before running the multi-model pipeline a second time.