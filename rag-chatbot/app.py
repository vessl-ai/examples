import argparse
import logging
import os
import shutil
import sys
from time import sleep
from typing import List, Optional

import gradio as gr
import torch

from chromadb.config import Settings as ChromaDBSettings
from langchain_chroma import Chroma
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def generate_text_chunks(pdf_doc_path: str):
    loader = PyMuPDFLoader(file_path=pdf_doc_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=512)
    text_chunks = text_splitter.split_documents(documents)

    logger.info(f"Parsed {len(text_chunks)} chunks from {pdf_doc_path}")
    return text_chunks

class RetrievalChain:
    def __init__(
        self,
        docs_folder: str,
        embedding_model_name: str,
        llm_model_name: str,
        llm_api_endpoint: str,
        llm_api_key: Optional[str],
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

    def initialize_chain(self, initial_docs: List[str]):
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

        # Turn off tokenizers parallelism to avoid deadlocks potentially caused by parallel chains
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Prompt
        prompt = ChatPromptTemplate.from_template(
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Keep the answer concise. If you don't know the answer, Explain there is not enough information to answer the question.\n"
            "Question: {question}\n"
            "Context: {context}]\n"
            "Answer:")

        # LLM
        llm = ChatOpenAI(
            base_url=self.llm_endpoint,
            model=self.llm_model_name,
            openai_api_key=self.llm_api_key or "na",
            streaming=True,
            temperature=0.5,
            max_tokens=4096,
        )

        # Conform chain
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: "\n\n".join(doc.page_content for doc in x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )

        # Connect chain with retriever
        self.chain = RunnableParallel(
            {"context": self.vector_store.as_retriever(), "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

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
        full_response = ""
        try:
            for response in self.chain.stream(message):
                if "answer" in response:
                    full_response += response["answer"]
                yield full_response
        except Exception as e:
            gr.Warning(f"Error while generating message:\n{e}")
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

    chain = RetrievalChain(
        docs_folder="./docs",
        embedding_model_name=args.embedding_model_name,
        llm_model_name=args.llm_model_name,
        llm_api_endpoint=args.llm_api_endpoint,
        llm_api_key=args.llm_api_key,
        chroma_server_host=args.chroma_server_host,
        chroma_server_http_port=args.chroma_server_http_port,
    )
    chain.initialize_chain(initial_docs)

    with gr.Blocks(css=css, title="🦜🔗 RAG Chatbot with LangChain and Open-source LLMs") as demo:
        with gr.Row():
            gr.Markdown(
            f"""<h2>RAG Chatbot with PDF documents</h2>
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
                        fn=chain.add_document, inputs=[document], outputs=[generate_vectordb_btn])
            with gr.Column(scale=2):
                gr.ChatInterface(chain.handle_chat)
        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
            close_button.click(fn=lambda: gr.update(interactive=False), outputs=[close_button]).then(fn=close_app)

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="🦜🔗 RAG Chatbot",
        description="Question Answering with LangChain and Chroma vector stores.")

    parser.add_argument("--port", default=7860, type=int, help="Port to run the Gradio server.")
    parser.add_argument("--docs-folder", default="./docs", help="Path to the folder containing the PDF documents.")
    parser.add_argument("--embedding-model-name", default="BAAI/bge-m3", help="HuggingFace model name for text embeddings.")
    parser.add_argument("--llm-model-name", default="casperhansen/llama-3-8b-instruct-awq", help="HuggingFace model name for LLM.")
    parser.add_argument("--llm-api-endpoint", default=None, help="OpenAI or compatible API endpoint.")
    parser.add_argument("--llm-api-key", default=None, help="API key for OpenAI-compatible LLM API.")
    parser.add_argument("--chroma-server-host", default=None, help="Chroma server host. If not provided, Chroma will run as in-memory ephemeral client.")
    parser.add_argument("--chroma-server-http-port", default=None, type=int, help="Chroma server HTTP port.")

    args = parser.parse_args()

    main(args)
