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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    text_chunks = text_splitter.split_documents(documents)

    logger.info(f"Parsed {len(text_chunks)} chunks from {pdf_doc_path}")
    return text_chunks

class RetrievalChain:
    def __init__(
        self,
        docs_folder: str,
        embedding_model_name: str,
        llm_model_name: str,
        llm_host: str,
        llm_api_key: Optional[str],
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
        chroma_collection_name: str = "rag-chatbot",
    ):
        self.docs_folder = docs_folder
        self.document_files = []
        self.document_chunk_count = 0
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"})
        self.llm_host = llm_host
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name

        chroma_client_settings = None
        if chroma_host:
            chroma_client_settings=ChromaDBSettings(
                chroma_host=chroma_host,
                chroma_port=chroma_port)
        self.vector_store = Chroma(
            chroma_collection_name,
            embedding_function=self.embeddings,
            client_settings=chroma_client_settings)

    def _docs_stat(self):
        return f"{len(self.document_files)} docs ({self.vector_store._collection.count()} chunks)"

    def _conform_chain(self):
        # Try LLM connection
        try:
            message = self.llm.invoke("Hello, checking you're up and running.")
            logger.info(f"Connected to LLM: {message}")
        except Exception as e:
            gr.Warning(f"Failed to connect to LLM endpoint {self.llm_host}: {e}, please check the connection.")

        # Conform chain
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: "\n\n".join(doc.page_content for doc in x["context"])))
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Connect chain with retriever
        self.chain = RunnableParallel(
            {"context": self.vector_store.as_retriever(), "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

    def initialize_chain(self, initial_docs: List[str]):
        """
        Initialize a vector database from a list of PDF documents.

        Parameters
        ----------
        docs_folder : str
            Path to the folder containing the PDF documents.

        """

        logger.info(f"Initializing vector database from {len(initial_docs)} Documents...")
        self.document_files = initial_docs
        for pdf_file_path in initial_docs:
            chunks = generate_text_chunks(pdf_file_path)
            self.vector_store.add_documents(chunks)

        # Turn off tokenizers parallelism to avoid deadlocks potentially caused by parallel chains
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Prompt
        self.prompt_str = """
        """
        self.prompt = ChatPromptTemplate.from_template(
            "As an AI assistant, you are assigned to perform question-answering tasks. Utilize the provided retrieved context to formulate your answer to the asked question. If the context doesn't contain sufficient information for a direct or comprehensive response, try to infer as much as possible from the provided details and draft a plausible answer. However, make sure to clarify that the given context didn't offer enough information to provide a fully accurate response.\n"
            "Presented Question: {question}\n"
            "Provided Context: {context}\n"
            "Your Response:")

        # LLM
        self.llm = ChatOpenAI(
            base_url=self.llm_host,
            model=self.llm_model_name,
            openai_api_key=self.llm_api_key or "na",
            streaming=True,
            temperature=0.5,
            max_tokens=4096,
        )
        self._conform_chain()

    def update_chain_config(self, llm_host: str, llm_api_key: str):
        self.llm_host = llm_host.strip()
        self.llm_api_key = llm_api_key.strip()
        self.llm = ChatOpenAI(
            base_url=self.llm_host,
            model=self.llm_model_name,
            openai_api_key=self.llm_api_key or "na",
            streaming=True,
            temperature=0.5,
            max_tokens=4096,
        )
        self._conform_chain()
        gr.Info("LLM settings updated!")
        return gr.update(value="Apply", interactive=True)

    def add_document(self, list_file_obj: List, progress=gr.Progress()):
        if self.vector_store is None:
            raise ValueError("Vectorstore not initialized. Please run initialize_database() first.")

        if list_file_obj is None:
            return gr.update(value="Upload PDF documents", interactive=True)

        pdf_docs = [x.name for x in list_file_obj if x is not None]
        for i, pdf in enumerate(pdf_docs):
            progress((i, len(pdf_docs)), desc=f"\nCopying {pdf.split('/')[-1]} to the docs folder\n")
            shutil.copy(pdf, self.docs_folder)

        try:
            current_progress = len(pdf_docs)
            total_progress = len(pdf_docs)
            for pdf_file_path in pdf_docs:
                self.document_files.append(pdf_file_path)
                nodes = generate_text_chunks(pdf_file_path)
                total_progress += len(nodes)
                for node in nodes:
                    # Add nodes one by one to the vector store to show progress
                    current_progress += 1
                    progress((current_progress, total_progress), desc=f"\nGenerating vector chunks from {pdf_file_path.split('/')[-1]}\n")
                    self.vector_store.add_documents([node])

            gr.Info("Upload Completed!")
        except Exception as e:
            gr.Warning(f"Upload failed: {e}")

        return [
            gr.update(label=self._docs_stat(), value=self.document_files), # documents_filebox
            gr.update(value=None),                                         # uploader
            gr.update(value="Upload PDF documents", interactive=True)      # upload_pdf_btn
        ]

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
    .contain { display: flex; flex-direction: column; }
    .gradio-container { height: 100vh !important; }
    #component-0, .tabs { height: 100%; }
    #chatbot-container { flex-grow: 1; overflow: auto; }

    .tabs, .tabitem .gap { height: 100%; }
    .tabitem { height: 95%; }
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
        llm_host=args.llm_host,
        llm_api_key=args.llm_api_key,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
    )
    chain.initialize_chain(initial_docs)

    with gr.Blocks(css=css, title="🦜🔗 RAG Chatbot with LangChain and Open-source LLMs") as demo:
        with gr.Row():
            gr.Markdown(
            """<h2>RAG Chatbot with PDF documents</h2>
            <h3>Ask any questions about your PDF documents, along with follow-ups<br>PDF 문서를 업로드하고, 문서에 대한 질문을 해보세요!</h3>
            This AI assistant performs retrieval-augmented generation from your PDF documents. You can add more documents by uploading PDF files one the left sidebar.<br>
            이 AI 어시스턴트는 PDF 문서에서 정보를 검색하여 답변을 생성합니다. 왼쪽 사이드바에서 PDF 파일을 업로드하여 더 많은 문서를 추가할 수 있습니다.<br>
            """)
        with gr.Tab("Chatbot"):
            with gr.Row(elem_id="chatbot-container"):
                with gr.Column(scale=1):
                    with gr.Row():
                        documents_filebox = gr.Files(label=chain._docs_stat(), value=initial_docs, interactive=False)
                    with gr.Row():
                        uploader = gr.Files(
                            file_count="multiple",
                            file_types=["pdf"],
                            interactive=True,
                            label="Upload your PDF documents here")
                    with gr.Row():
                        upload_pdf_btn = gr.Button("Upload PDF documents")
                        upload_pdf_btn.click(
                            fn=lambda: gr.update(value="Uploading...", interactive=False), outputs=[upload_pdf_btn]
                        ).then(
                            fn=chain.add_document, inputs=[uploader], outputs=[documents_filebox, uploader, upload_pdf_btn])
                with gr.Column(scale=3):
                    gr.ChatInterface(chain.handle_chat)
        with gr.Tab("Settings"):
            with gr.Group():
                llm_host_textbox = gr.Textbox(args.llm_host, label="LLM Host")
                llm_api_key_textbox = gr.Textbox(args.llm_api_key, label="LLM API Key")
            with gr.Group():
                update_llm_btn = gr.Button("Apply")
                update_llm_btn.click(
                    fn=lambda: gr.update(value="Connecting...", interactive=False), outputs=[update_llm_btn]
                ).then(
                    fn=chain.update_chain_config, inputs=[llm_host_textbox, llm_api_key_textbox], outputs=[update_llm_btn])
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
    parser.add_argument("--llm-host", default="http://localhost:8000/v1", help="OpenAI or compatible API endpoint.")
    parser.add_argument("--llm-api-key", default=None, help="API key for OpenAI-compatible LLM API.")
    parser.add_argument("--chroma-host", default=None, help="Chroma server host. If not provided, Chroma will run as in-memory ephemeral client.")
    parser.add_argument("--chroma-port", default=8000, type=int, help="Chroma server HTTP port.")

    args = parser.parse_args()

    main(args)
