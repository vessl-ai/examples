import argparse
import os
import shutil
from time import sleep
from typing import Any, List

import gradio as gr
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import VLLM
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import torch

def get_pdf_text(pdf_docs):
    """
    Extract text from a list of PDF documents.

    Parameters
    ----------
    pdf_docs : list
        List of PDF documents to extract text from.

    Returns
    -------
    str
        Extracted text from all the PDF documents.

    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Split the input text into chunks.

    Parameters
    ----------
    text : str
        The input text to be split.

    Returns
    -------
    list
        List of text chunks.

    """
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1500, chunk_overlap=300, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

class RAGInterface:
    def __init__(
        self,
        embedding_model_name: str,
        encode_kwargs: dict,
        docs_folder: str = "./docs",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name, encode_kwargs=encode_kwargs, model_kwargs={"device": self.device}
        )
        self.encode_kwargs = encode_kwargs
        self.vectorstore: FAISS = None
        self.docs_folder = docs_folder
        print(f"Using accelerator: {self.device}")

    def initialize_conversation_chain(self, initial_docs: List[str], model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize a vector database from a list of PDF documents.

        Parameters
        ----------
        docs_folder : str
            Path to the folder containing the PDF documents.

        """

        print(f"Scanning all pdf files in {len(initial_docs)} Documents...")

        raw_text = get_pdf_text(initial_docs)
        if raw_text == "":
            raw_text = "Initial text"
        text_chunks = get_text_chunks(raw_text)

        print("Initializing vector database...")
        self.vectorstore = FAISS.from_texts(texts=text_chunks, embedding=self.embedding_model)

        print("Initializing conversation chain...")
        # llm = HuggingFaceHub(
        #     repo_id=model_name,
        #     model_kwargs={"temperature": 0.5, "max_length": 4096, "device": self.device},
        # )
        llm = VLLM(
            model=model_name,
            trust_remote_code=True,  # mandatory for hf models
            max_new_tokens=2048,
            top_k=10,
            top_p=0.95,
            temperature=0.8,
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=self.vectorstore.as_retriever(), memory=memory
        )

    def add_document(self, list_file_obj: List, progress=gr.Progress()):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Please run initialize_database() first.")

        if list_file_obj is None:
            return gr.update(value="Upload PDF documents", interactive=True)

        gr.Info(f"Adding {len(list_file_obj)} documents to vector database...")
        progress(0, desc="Copying documents to ")
        pdf_docs = [x.name for x in list_file_obj if x is not None]
        for pdf in pdf_docs:
            shutil.copy(pdf, self.docs_folder)
        gr.Info("Extracting text from PDFs...")
        raw_text = get_pdf_text(pdf_docs)
        gr.Info("Splitting text into chunks...")
        text_chunks = get_text_chunks(raw_text)
        gr.Info("Adding chunks to vector database...")
        self.vectorstore.add_texts(texts=text_chunks)

        gr.Info("Upload Completed!")
        return gr.update(value="Upload PDF documents", interactive=True)

    def get_retriever(self):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Please run initialize_database() first.")
        return self.vectorstore.as_retriever()

    def handle_chat(self, message, history):
        full_reponse = ""
        for response in self.conversation.stream(message):
            print(response)
            full_reponse += response
            yield full_reponse

def close_app():
    gr.Info("Terminated the app!")
    sleep(1)
    os._exit(0)


def main(args: argparse.Namespace):
    if os.environ.get("HUGGINGFACEHUB_API_TOKEN", "") == "" and args.hf_token != "":
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = args.hf_token

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
        embedding_model_name=args.embedding_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    ragger.initialize_conversation_chain(initial_docs, model_name=args.model_name)

    with gr.Blocks(css=css, title="PDF Chatbot with LangChain🦜 and Open-source LLMs") as demo:
        with gr.Row():
            gr.Markdown(
            f"""<h2>PDF Chatbot with LangChain🦜 and {args.model_name}</h2>
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
        prog="PDF Chatbot",
        description="Question Answering with Retrieval QA and LangChain Language Models featuring FAISS vector stores.")

    parser.add_argument("--docs-folder", default="./docs")
    parser.add_argument("--embedding-model-name", default="BAAI/bge-m3")
    parser.add_argument("--model-name", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--hf-token", default="")

    args = parser.parse_args()

    main(args)
