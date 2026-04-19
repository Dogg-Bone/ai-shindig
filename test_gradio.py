import gradio as gr

def chat(message, history):
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "Hello"})
    return "", history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    msg.submit(chat, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    print("Gradio version:", gr.__version__)
