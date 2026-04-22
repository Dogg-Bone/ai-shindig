import json
import logging
import re
from typing import List, Dict, Any, TypedDict, Annotated, Optional
import operator

from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ddgs import DDGS
from langgraph.graph import StateGraph, START, END

# Setup logging
logging.basicConfig(level=logging.INFO, format='\033[90m%(asctime)s - [%(levelname)s] - %(message)s\033[0m')
logger = logging.getLogger('orchestrator')

# ==========================================
# Circuit Breaker Globals
# ==========================================
CONSECUTIVE_WEB_FAILURES = 0

# ==========================================
# 1. State Definition
# ==========================================
class AgentState(TypedDict):
    """
    Defines the shared memory structure passed between LangGraph nodes.
    Each node receives this state, performs its task, and returns an updated dictionary
    that merges into this state.
    """
    original_query: str
    sub_queries: List[str]
    raw_data: List[str]
    synthesized_findings: str
    confidence: float
    final_report: str
    phase: str  # Tracks the current step so the Manager node can route correctly

# ==========================================
# 2. Global Initialization (Embeddings & DB)
# ==========================================
logger.info("Initializing global embedding model and vector store...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./my_chroma_db_new", embedding_function=embedding_model, collection_name="documents")

# ==========================================
# 3. Tool Definitions
# ==========================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
def _do_web_search(query: str) -> str:
    # simulate internet outage
    if False:
        raise Exception("Simulated internet outage.")
    results = DDGS().text(query, max_results=3)
    if not results:
        return ""
    snippets = [f"Source: {res.get('href', 'Unknown')}\nSnippet: {res.get('body', '')}" for res in results]
    return "\n\n".join(snippets)

def web_search_tool(query: str) -> str:
    """
    Hits an external search API (DuckDuckGo) and returns the top 3 snippets.
    This acts as the agent's window to the live internet.
    """
    global CONSECUTIVE_WEB_FAILURES

    if CONSECUTIVE_WEB_FAILURES >= 5:
        logger.warning(f"Circuit breaker tripped. Skipping web search for: '{query}'")
        return "Service Temporarily Unavailable"

    logger.info(f"Executing web search for: '{query}'")
    try:
        result_text = _do_web_search(query)
        CONSECUTIVE_WEB_FAILURES = 0 # reset on success
        if not result_text:
            logger.info(f"No web search results found for: '{query}'")
            return f"No web search results found for '{query}'."
        return result_text
    except Exception as e:
        CONSECUTIVE_WEB_FAILURES += 1
        logger.error(f"Web search failed after retries: {e}. Consecutive failures: {CONSECUTIVE_WEB_FAILURES}")
        return f"Web search failed for '{query}'."

def document_retrieval_tool(query: str) -> str:
    """
    Searches the local ChromaDB vector store for the query and returns the top 3
    most relevant chunks of text along with their similarity scores.
    This acts as the agent's internal knowledge base.
    """
    logger.info(f"Executing document retrieval for: '{query}'")
    try:
        # Retrieve top 3 chunks with their distance scores
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)
        if not docs_and_scores:
            logger.info(f"No internal documents found for: '{query}'")
            return f"No internal documents found for '{query}'."

        snippets = [f"Snippet from document (Metadata: {doc.metadata}, Distance: {score:.4f}):\n{doc.page_content}" for doc, score in docs_and_scores]
        return "\n\n".join(snippets)
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return f"Document retrieval failed for '{query}'."


# ==========================================
# 3.1 Sanitization Utility
# ==========================================
def sanitize_web_content(text: str) -> str:
    """
    Strips HTML tags, markdown code blocks, and system-level prompt words
    to mitigate prompt injection risks.
    """
    if not text:
        return text

    # Strip HTML tags
    text = re.sub(r'<[^>]*>', '', text)

    # Strip markdown code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # Strip dangerous system-level prompt words case-insensitively
    dangerous_phrases = [
        r"ignore previous instructions",
        r"system:",
        r"assistant:",
        r"user:"
    ]
    for phrase in dangerous_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)

    return text.strip()

# ==========================================
# 4. Model Setup
# ==========================================
# Initialize the connection to our local inference server (model_server.py).
# We use the ChatOpenAI class but point its base URL to our local FastAPI app.
# This simulates hitting a cloud endpoint but runs entirely locally.
local_llm = ChatOpenAI(
    model="Qwen/Qwen2.5-3B-Instruct",
    openai_api_base="http://0.0.0.0:8000/v1",
    openai_api_key="sk-no-key-required",
    max_tokens=1000,
    temperature=0.1
)

# ==========================================
# 5. Agent Nodes
# ==========================================

def decomposer_node(state: AgentState) -> AgentState:
    """
    Takes a complex user question and breaks it down into smaller, searchable sub-queries.
    This is necessary because standard search algorithms struggle with multi-part questions.
    """
    logger.info("--- Decomposer Agent Executing ---")
    query = state["original_query"]

    # Prompting the LLM to output ONLY a JSON array
    prompt = f"""You are a Decomposer Agent. Your task is to break down the following complex user question into an array of 2 to 4 simple, independent search queries.
Output ONLY a raw JSON list of strings representing the sub-queries. Do not include markdown blocks like ```json or any other text.

Question: {query}
"""

    response = local_llm.invoke([SystemMessage(content="You are a JSON-only response agent. Only output a valid JSON list of strings."), HumanMessage(content=prompt)])

    # Parse the LLM's response safely
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        if content.startswith("```"):
            content = content[3:-3]

        sub_queries = json.loads(content)
        if not isinstance(sub_queries, list):
             sub_queries = [str(query)]
    except Exception as e:
        logger.error(f"Failed to parse Decomposer output as JSON: {response.content}")
        # Fallback to the original query as the only sub-query
        sub_queries = [query]

    logger.info(f"Decomposer Output: {sub_queries}")
    return {"sub_queries": sub_queries, "phase": "search"}


def search_agent_node(state: AgentState) -> AgentState:
    """
    Iterates over the decomposed sub-queries, executes the tools (Web Search & DB Search),
    and uses the LLM to summarize the findings for each sub-query.
    """
    logger.info("--- Search Agent Executing ---")
    sub_queries = state["sub_queries"]
    raw_data_collected = []

    # Prompt template for summarizing the raw tool outputs
    prompt_template = """You are a Search/Retrieval Agent. Your task is to process the following sub-query by evaluating the retrieved context from the Web and Internal Documents.
The Internal Documents Context includes distance scores alongside the metadata for each snippet. A lower distance score indicates higher statistical relevance to the query.

Sub-Query: {sq}

-- Web Search Context --
{web_res}

-- Internal Documents Context --
{doc_res}

Based on the context above, provide a brief, factual summary of the information relevant to the sub-query.
Also, assign a confidence score (float between 0.0 and 1.0) to your summary. Use the relevance (e.g. distance scores where lower is better) of the retrieved context to inform your confidence.
Output ONLY a JSON object containing two keys:
1. "summary" (string): your factual summary.
2. "confidence_score" (float): your confidence score.
Do not include markdown blocks like ```json or any other text.
"""

    for sq in sub_queries:
        logger.info(f"Search Agent processing sub-query: '{sq}'")
        # Execute tools
        web_res = web_search_tool(sq)
        web_res = sanitize_web_content(web_res)
        doc_res = document_retrieval_tool(sq)

        # Check if both searches failed/found nothing
        if ("No web search results found" in web_res or "failed" in web_res or "Service Temporarily Unavailable" in web_res) and \
           ("No internal documents found" in doc_res or "failed" in doc_res):
            logger.info(f"No results found for sub-query: '{sq}'. Setting confidence to 0.0.")
            combined_res = f"=== Summary for sub-query: '{sq}' (Confidence: 0.0) ===\nNo information found."
            raw_data_collected.append(combined_res)
            continue

        # Format prompt with tool results
        prompt = prompt_template.format(sq=sq, web_res=web_res, doc_res=doc_res)

        # Invoke LLM to process and summarize the search results
        response = local_llm.invoke([
            SystemMessage(content="You are a JSON-only response agent. Only output a valid JSON object."),
            HumanMessage(content=prompt)
        ])

        summary_text = "Summary extraction failed."
        conf_score = 0.0
        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            if content.startswith("```"):
                content = content[3:-3]

            res_json = json.loads(content)
            if isinstance(res_json, dict):
                summary_text = res_json.get("summary", "No summary provided.")
                try:
                    conf_score = float(res_json.get("confidence_score", 0.0))
                except ValueError:
                    conf_score = 0.0
        except Exception as e:
            logger.error(f"Failed to parse Search Agent output as JSON: {response.content}")

        combined_res = f"=== Summary for sub-query: '{sq}' (Confidence: {conf_score}) ===\n{summary_text}\n"
        raw_data_collected.append(combined_res)

    logger.info(f"Search Agent collected {len(raw_data_collected)} sets of summarized data.")
    return {"raw_data": raw_data_collected, "phase": "synthesis"}


def synthesizer_node(state: AgentState) -> AgentState:
    """
    Takes all the individual summaries from the Search Agent and synthesizes them
    into a single cohesive set of findings, resolving any minor conflicts.
    """
    logger.info("--- Synthesizer Agent Executing ---")
    raw_data_str = "\n".join(state["raw_data"])

    prompt = f"""You are an Analyst Agent. Review the raw data provided below, filter out irrelevant noise, resolve minor conflicts, and summarize the key findings.
Note that the individual summaries in the Raw Data each have an attached confidence score based on the statistical relevance of their source material. You should use these individual confidence scores to weight the synthesized items and properly inform your final confidence score.

Raw Data:
{raw_data_str}

Output ONLY a JSON object containing two keys:
1. "synthesized_findings" (string): a concise, synthesized summary of facts based ONLY on the provided data.
2. "confidence_score" (float): a value between 0.0 and 1.0 indicating your overall confidence in the synthesized findings, weighted by the confidence scores of the raw data.
Do not include markdown blocks like ```json or any other text.
"""

    response = local_llm.invoke([SystemMessage(content="You are a JSON-only response agent. Only output a valid JSON object."), HumanMessage(content=prompt)])

    # Parse JSON safely
    synthesized_findings = "Synthesis failed."
    confidence_score = 0.0
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        if content.startswith("```"):
            content = content[3:-3]

        result_json = json.loads(content)
        if isinstance(result_json, dict):
            synthesized_findings = result_json.get("synthesized_findings", "No findings provided.")
            try:
                confidence_score = float(result_json.get("confidence_score", 0.0))
            except ValueError:
                confidence_score = 0.0
    except Exception as e:
        logger.error(f"Failed to parse Synthesizer output as JSON: {response.content}")

    logger.info(f"Synthesizer completed analysis with confidence {confidence_score}.")
    return {"synthesized_findings": synthesized_findings, "confidence": confidence_score, "phase": "final_report"}


def manager_node(state: AgentState) -> AgentState:
    """
    The orchestrator. It evaluates the current state of the research process.
    If the process just started, it routes to the Decomposer.
    If the synthesis is complete, it drafts the final report and routes to END.
    """
    logger.info("--- Manager Agent Executing ---")

    phase = state.get("phase", "init")

    if phase == "init":
        # State just initialized. Need to kick off the pipeline.
        logger.info("Manager evaluating query: Routing to Decomposer.")
        return {"phase": "decompose"}

    elif phase == "final_report":
        confidence = state.get("confidence", 0.0)
        if confidence < 0.6:
            logger.warning(f"Low confidence detected ({confidence}). Prompting user.")
            print(f"\n[WARNING] The system has low confidence ({confidence}) in its findings.")
            while True:
                choice = input("Do you want to (p)roceed, (r)estart, or (q)uit? [p/r/q]: ").strip().lower()
                if choice in ['p', 'r', 'q']:
                    break
                print("Invalid choice. Please enter 'p', 'r', or 'q'.")

            if choice == 'r':
                logger.info("User chose to restart. Resetting state.")
                return {"sub_queries": [], "raw_data": [], "synthesized_findings": "", "confidence": 0.0, "final_report": "", "phase": "decompose"}
            elif choice == 'q':
                logger.info("User chose to quit.")
                print("\nAborting current query process.")
                return {"sub_queries": [], "raw_data": [], "synthesized_findings": "", "confidence": 0.0, "final_report": "User aborted due to low confidence.", "phase": "end"}

        # The synthesizer has finished, so we can now draft the final answer.
        logger.info("Manager drafting final report.")

        prompt = f"""You are a Project Manager. Draft a final, comprehensive report answering the user's original query based on the synthesized findings.

Original Query: {state['original_query']}

Synthesized Findings:
{state['synthesized_findings']}

Provide a clear, well-structured final answer.
"""
        response = local_llm.invoke([SystemMessage(content="You are a Project Manager drafting a final report."), HumanMessage(content=prompt)])

        return {"final_report": response.content, "phase": "end"}

    return state


# ==========================================
# 6. Graph Routing Logic
# ==========================================
def router(state: AgentState) -> str:
    """
    The conditional routing function for the Manager node.
    It reads the 'phase' set by the previous nodes and determines which node to run next.
    """
    phase = state.get("phase")
    if phase == "decompose":
        return "decomposer"
    elif phase == "search":
        return "search_agent"
    elif phase == "synthesis":
        return "synthesizer"
    elif phase == "end":
        return END
    else:
        return END # Failsafe

# ==========================================
# 7. Graph Compilation
# ==========================================
# Create the LangGraph StateGraph object with our AgentState definition
workflow = StateGraph(AgentState)

# Add all our agent functions as nodes in the graph
workflow.add_node("manager", manager_node)
workflow.add_node("decomposer", decomposer_node)
workflow.add_node("search_agent", search_agent_node)
workflow.add_node("synthesizer", synthesizer_node)

# Define the flow/edges between the nodes:
# START -> manager
workflow.add_edge(START, "manager")

# manager -> (dynamic routing based on the router function)
workflow.add_conditional_edges("manager", router)

# decomposer -> search_agent -> synthesizer -> manager
workflow.add_edge("decomposer", "search_agent")
workflow.add_edge("search_agent", "synthesizer")
workflow.add_edge("synthesizer", "manager")

# Compile the graph into a runnable application
app = workflow.compile()

# ==========================================
# 8. Execution and Testing
# ==========================================
def run_orchestrator():
    print("\n" + "="*50)
    print("Welcome to the Orchestrator-Worker Research Assistant")
    print("Make sure your model server is running! (uvicorn model_server:app --host 0.0.0.0 --port 8000)")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\nEnter your question (or type 'quit' to exit):\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if user_input.strip() == "":
                print("Please enter a valid question.")
                continue

            if len(user_input) > 1000:
                print("Question too long. Please keep it under 1000 characters.")
                continue

            initial_state = {
                "original_query": sanitize_web_content(user_input),
                "sub_queries": [],
                "raw_data": [],
                "synthesized_findings": "",
                "final_report": "",
                "phase": "init"
            }

            print("\n--- Processing via LangGraph ---")

            for event in app.stream(initial_state, stream_mode="values"):
                # values yields the full state after each node
                pass

            # The final state will be the last event
            final_state = event

            print("\n" + "="*50)
            print("FINAL REPORT:")
            print("="*50)
            print(final_state["final_report"])
            print("="*50 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            break

if __name__ == "__main__":
    run_orchestrator()
