import gradio as gr
from pipeline import run_pipeline
import logging

logger = logging.getLogger('rag_pipeline')

def chat_interface(message, history):
    # We need to return the new history and the details panel content

    try:
        result = run_pipeline({"question": message})
    except Exception as e:
        logger.error(f"Error during run_pipeline: {e}")
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"An error occurred: {str(e)}"}], "Error generating details."

    if result["status"] == "success":
        output = result["output"]
        answer = output.answer

        details = f"**Processing Time:** {output.processing_time_ms} ms\n\n**Sources:**\n"
        for source in output.sources:
            details += f"- {source}\n"

        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}], details

    elif result["status"] == "skipped_due_to_low_relevance":
        answer = "I couldn't find sufficiently relevant information to answer your question. Please review the retrieved documents in the details panel."

        details = "**POLICY ENFORCEMENT: LOW RELEVANCE**\n\nThe system did not find sufficiently relevant information. Here are the top retrieved documents:\n\n"

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

            details += f"**--- Document {i+1} ---**\n"
            details += f"**Citation:** [{source_info}, Page {page}, Paragraph {paragraph}]\n"
            details += f"**Relevance Score:** {score:.2f}\n"
            details += f"**Content:** {doc}\n\n"

        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}], details
    else:
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Unknown status returned from pipeline."}], "No details available."

with gr.Blocks(title="Reality TV RAG") as demo:
    gr.Markdown("# Reality TV RAG Chat")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=600)
            msg = gr.Textbox(label="Your Question", placeholder="Enter your question here...")
            clear = gr.Button("Clear Chat")
        with gr.Column(scale=1):
            details_panel = gr.Markdown(label="Details", value="Sources and processing time will appear here.")

    def user_input(user_message, chat_history):
        # Return updated history and details
        updated_history, details = chat_interface(user_message, chat_history)
        return "", updated_history, details

    msg.submit(user_input, [msg, chatbot], [msg, chatbot, details_panel])
    clear.click(lambda: ([], "Sources and processing time will appear here."), None, [chatbot, details_panel])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
