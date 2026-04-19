with open("ui_pipeline.py", "r") as f:
    content = f.read()

search_str = """def chat_interface(message, history):
    # We need to return the new history and the details panel content

    try:
        result = run_pipeline({"question": message})
    except Exception as e:
        logger.error(f"Error during run_pipeline: {e}")
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"An error occurred: {str(e)}"}], "Error generating details."

    if result["status"] == "success":
        output = result["output"]
        answer = output.answer

        details = f"**Processing Time:** {output.processing_time_ms} ms\\n\\n**Sources:**\\n"
        for source in output.sources:
            details += f"- {source}\\n"

        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}], details

    elif result["status"] == "skipped_due_to_low_relevance":
        answer = "I couldn't find sufficiently relevant information to answer your question. Please review the retrieved documents in the details panel."

        details = "**POLICY ENFORCEMENT: LOW RELEVANCE**\\n\\nThe system did not find sufficiently relevant information. Here are the top retrieved documents:\\n\\n"

        # We expect pipeline.py to return docs, metadatas, and scores.
        docs = result.get("docs", [])
        metadatas = result.get("metadatas", [])
        scores = result.get("scores", [])

        for i, doc in enumerate(docs):
            meta = metadatas[i] if i < len(metadatas) else {}
            score = scores[i] if i < len(scores) else 0.0
            source_info = meta.get('source', 'Unknown')
            page = meta.get('page', 'Unknown')
            paragraph = meta.get('paragraph', 'Unknown')

            details += f"**--- Document {i+1} ---**\\n"
            details += f"**Citation:** [{source_info}, Page {page}, Paragraph {paragraph}]\\n"
            details += f"**Relevance Score:** {score:.2f}\\n"
            details += f"**Content:** {doc}\\n\\n"

        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}], details
    else:
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Unknown status returned from pipeline."}], "No details available."

with gr.Blocks(title="Multi-Agent RAG Pipeline") as demo:
    gr.Markdown("# Multi-Agent RAG Pipeline Chat")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=600, type="messages")
            msg = gr.Textbox(label="Your Question", placeholder="Enter your question here...")"""

# First read and check what is actually there because we previously used tuple format!
