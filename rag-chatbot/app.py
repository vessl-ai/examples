import argparse
import logging
import os
import shutil
from time import sleep
from typing import List, Optional

import gradio as gr
import torch

from chromadb.config import Settings as ChromaDBSettings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI


logger = logging.getLogger(__name__)

def generate_text_chunks(pdf_doc_path: str):
    loader = PyMuPDFLoader(file_path=pdf_doc_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=512)
    text_chunks = text_splitter.split_documents(documents)

    logger.info(f"Parsed {len(text_chunks)} chunks from {pdf_doc_path}")
    return text_chunks

class RAGInterface:
    def __init__(
        self,
        docs_folder: str,
        embedding_model_name: str,
        llm_api_endpoint: str,
        llm_api_key: Optional[str],
        llm_model_name: str,
        chroma_server_host: Optional[str] = None,
        chroma_server_http_port: Optional[int] = None,
        chroma_collection_name: str = "rag-chatbot",
    ):
        self.docs_folder = docs_folder
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})
        self.llm_endpoint = llm_api_endpoint
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name

        chroma_client_settings = None
        if chroma_server_host:
            chroma_client_settings=ChromaDBSettings(
                chroma_server_host=chroma_server_host,
                chroma_server_http_port=chroma_server_http_port)
        self.vector_store = Chroma(
            chroma_collection_name,
            embedding_function=self.embeddings,
            client_settings=chroma_client_settings)

    def initialize_chat_engine(self, initial_docs: List[str]):
        """
        Initialize a vector database from a list of PDF documents.

        Parameters
        ----------
        docs_folder : str
            Path to the folder containing the PDF documents.

        """

        logger.info(f"Initializing vector database from {len(initial_docs)} Documents...")
        for pdf_file_path in initial_docs:
            chunks = generate_text_chunks(pdf_file_path)
            self.vector_store.add_documents(chunks)

        logger.info("Initializing conversation chain...")
        llm = ChatOpenAI(
            base_url=self.llm_endpoint,
            openai_api_key=self.llm_api_key,
            model_name=self.llm_model_name,
            streaming=True,
            temperature=0.5,
            max_tokens=4096,
        )
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chat_engine = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=self.vector_store.as_retriever(), memory=memory
        )

    def add_document(self, list_file_obj: List, progress=gr.Progress()):
        if self.vector_store is None:
            raise ValueError("Vectorstore not initialized. Please run initialize_database() first.")

        if list_file_obj is None:
            return gr.update(value="Upload PDF documents", interactive=True)

        gr.Info(f"Adding {len(list_file_obj)} documents to vector database...")
        progress(0, desc="Copying documents to ")
        pdf_docs = [x.name for x in list_file_obj if x is not None]
        for pdf in pdf_docs:
            shutil.copy(pdf, self.docs_folder)

        try:
            gr.Info("Adding documents into vector database...")
            for pdf_file_path in pdf_docs:
                nodes = generate_text_chunks(pdf_file_path)
                self.vector_store.add_documents(nodes)
                progress(1, desc=f"Adding {pdf_file_path} to vector database")

            gr.Info("Upload Completed!")
        except Exception as e:
            gr.Warning(f"Upload failed: {e}")
        return gr.update(value="Upload PDF documents", interactive=True)

    def handle_chat(self, message, history):
        streaming_response = self.chat_engine.stream(message)
        full_response = ""
        for token in streaming_response:
            print(token)
            yield full_response
        return full_response

def close_app():
    gr.Info("Terminated the app!")
    sleep(1)
    os._exit(0)


def main(args: argparse.Namespace):
    css = """
    footer {visibility: hidden}
    .toast-body.error {visibility: hidden}
    """

    initial_docs = []
    for root, _, files in os.walk(args.docs_folder):
        for file in files:
            if file.endswith(".pdf"):
                initial_docs.append(os.path.join(root, file))

    ragger = RAGInterface(
        docs_folder="./docs",
        embedding_model_name=args.embedding_model_name,
        llm_api_endpoint=args.llm_api_endpoint,
        llm_api_key=None,
        llm_model_name=args.model_name,
        chroma_server_host=args.chroma_server_host,
        chroma_server_http_port=args.chroma_server_http_port,
    )
    ragger.initialize_chat_engine(initial_docs)

    with gr.Blocks(css=css, title="RAG Chatbot with LlamaIndex🦙 and Open-source LLMs") as demo:
        with gr.Row():
            gr.Markdown(
            f"""<h2>RAG Chatbot with LlamaIndex🦙 and {args.model_name}</h2>
            <h3>Ask any questions about your PDF documents, along with follow-ups</h3>
            <b>Note:</b> This AI assistant performs retrieval-augmented generation from your PDF documents.<br>
            Initial documents are loaded from the `{args.docs_folder}` folder. You can add more documents by clicking the button below.<br>
            """)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    gr.Files(label="Initial documents", value=initial_docs, interactive=False)
                with gr.Row():
                    document = gr.Files(
                        file_count="multiple",
                        file_types=["pdf"],
                        interactive=True,
                        label="Add your PDF documents (single or multiple)")
                with gr.Row():
                    generate_vectordb_btn = gr.Button("Upload PDF documents")
                    generate_vectordb_btn.click(
                        fn=lambda: gr.update(value="Uploading...", interactive=False), outputs=[generate_vectordb_btn]
                    ).then(
                        fn=ragger.add_document, inputs=[document], outputs=[generate_vectordb_btn])
            with gr.Column(scale=2):
                gr.ChatInterface(ragger.handle_chat)
        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
            close_button.click(fn=lambda: gr.update(interactive=False), outputs=[close_button]).then(fn=close_app)

    demo.queue().launch(server_name="0.0.0.0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="RAG Chatbot",
        description="Question Answering with LangChain and Chroma vector stores.")

    parser.add_argument("--docs-folder", default="./docs", help="Path to the folder containing the PDF documents.")
    parser.add_argument("--embedding-model-name", default="BAAI/bge-m3", help="HuggingFace model name for text embeddings.")
    parser.add_argument("--llm_api_endpoint", default="https://run-execution-l96uwyig3uzm-run-execution-8080.oregon.google-cluster.vessl.ai/v1", help="OpenAI-compatible API endpoint.")
    parser.add_argument("--llm-model-name", default="TheBloke/Mistral-7B-Instruct-v0.2-AWQ", help="HuggingFace model name for LLM.")
    parser.add_argument("--llm_api_key", default=None, help="API key for OpenAI-compatible LLM API.")
    parser.add_argument("--chroma-server-host", default=None, help="Chroma server host. If not provided, Chroma will run as in-memory ephemeral client.")
    parser.add_argument("--chroma-server-http-port", default=None, type=int, help="Chroma server HTTP port.")

    args = parser.parse_args()

    main(args)
