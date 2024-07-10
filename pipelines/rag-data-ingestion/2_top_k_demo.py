import argparse
import os
import signal
from time import sleep

import gradio as gr

from util import get_embedding_model, get_chroma_vector_store

class ChromaRetriever:
    def __init__(
        self,
        chroma_endpoint: str,
        chroma_port: int,
        chroma_collection_name: str,
        embedding_model_name: str,
        top_k: int = 5):

        self.embedding_model = get_embedding_model(embedding_model_name)
        self.vector_store = get_chroma_vector_store(
            chroma_endpoint=chroma_endpoint,
            chroma_port=chroma_port,
            chroma_collection=chroma_collection_name,
            embedding_model=self.embedding_model)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})

    def handle_chat(self, message, history):
        # Retrieve nodes from the vector store

        retrieved_nodes = self.retriever.invoke(message)
        return "Retrieved Nodes: \n\n" + "\n".join([f"{node.metadata}: {node}" for node in retrieved_nodes])

def close_app():
    gr.Info("Terminated the app!")
    sleep(1)
    os._exit(0)

def timeout_handler(signum, frame):
    gr.Info("Timeout for 15 minutes - closing the app")
    close_app()

def main(args: argparse.Namespace):
    css = """
    footer {visibility: hidden}
    .toast-body.error {visibility: hidden}
    .chat-container {
        height: 70vh;
    }
    """

    retriever = ChromaRetriever(
        chroma_endpoint=args.chroma_endpoint,
        chroma_port=args.chroma_port,
        chroma_collection_name=args.chroma_collection_name,
        embedding_model_name=args.embedding_model,
        top_k=args.top_k,
    )

    with gr.Blocks(css=css, title="🥇🥈🥉 Top-k retrieval demo") as demo:
        with gr.Row():
            gr.Markdown(
                "## 🥇🥈🥉 Top-k retrieval test\n"
                "* Type your query in the chat box below to retrieve the top {args.top_k} results from the vector store.\n"
                "* The retrieved results will be displayed in the chat box.\n"
                "* Click the 'Close the app' button to terminate the app.\n"
                "* The app will automatically close after 20 minutes of inactivity.")
        with gr.Row():
            with gr.Column(elem_classes=["chat-container"]):
                gr.ChatInterface(retriever.handle_chat)
        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
            close_button.click(fn=lambda: gr.update(interactive=False), outputs=[close_button]).then(fn=close_app)

    # Set app timeout for 20 minutes
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(20 * 60)

    demo.queue().launch(server_name="0.0.0.0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Top-k retrieval demo",
        description="Top-k retrieval test demo using LlamaIndex and Gradio")

    parser.add_argument("--chroma-endpoint", type=str)
    parser.add_argument("--chroma-port", type=int, default=0)
    parser.add_argument("--chroma-collection-name", type=str, default="rag")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    main(args)
