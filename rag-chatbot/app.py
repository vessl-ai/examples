import argparse
import os
import shutil
import tqdm
from time import sleep
from typing import Dict, Any, List, Optional

import faiss
import gradio as gr

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.faiss import FaissVectorStore

import torch

CHAT_TEMPLATE = """
{%- set ns = namespace(first_system=false) -%}
{{-'<s>'-}}
{%- for message in messages %}
    {%- if message['role'] == 'system' -%}
        {{-' [INST] ' + message['content'] + '\n\n'-}}
        {%- set ns.first_system = true -%}
    {%- else -%}
        {%- if message['role'] == 'user' -%}
            {%- if ns.first_system -%}
                {{-'' + message['content'].rstrip() + ' [/INST] '-}}
                {%- set ns.first_system = false -%}
            {%- else -%}
                {{-' [INST] ' + message['content'].rstrip() + ' [/INST] '-}}
            {%- endif -%}
        {%- else -%}
            {{-'' + message['content'] + '</s>' -}}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{-''-}}
{%- endif -%}
"""


def generate_vector_store_nodes(pdf_doc_path: str, embed_model: HuggingFaceEmbedding):
    loader = PyMuPDFReader()
    documents = loader.load(file_path=pdf_doc_path)
    text_parser = SentenceSplitter(chunk_size=1024)

    text_chunks = []
    # maintain relationship with source doc index, to help inject doc metadata later
    doc_indices = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_indices.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_indices[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    for node in nodes:
        node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding

    print(f"Embedded {len(nodes)} nodes from {pdf_doc_path}")
    return nodes


class RAGInterface:
    def __init__(
            self,
            embedding_model_name: str,
            docs_folder: str = "./docs",
            stream: bool = False,
            use_flash_attention: bool = False,
            use_vllm: bool = True,
            vllm_kwargs: Optional[Dict[str, Any]] = {},
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name, device=self.device)

        # FAISS vector store
        self.faiss_index = faiss.IndexFlatL2(1024)  # 1024 is dimension of the embeddings
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)

        # LlamaIndex Storage context to store nodes mapped to vector store
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.vector_store_index = VectorStoreIndex(
            nodes=[], embed_model=self.embed_model, storage_context=self.storage_context)

        self.docs_folder = docs_folder
        self.stream = stream
        self.use_flash_attention = use_flash_attention
        self.use_vllm = use_vllm
        self.vllm_kwargs = vllm_kwargs if use_vllm else {}
        print(f"Using accelerator: {self.device}")

    def initialize_chat_engine(self, initial_docs: List[str], model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize a vector database from a list of PDF documents.

        Parameters
        ----------
        initial_docs: str
            Path to the folder containing the PDF documents.
        model_name: str
            Model name from huggingface.co/models in default or vllm if use_vllm is set to True
        """

        print(f"Initializing vector database from {len(initial_docs)} Documents...")
        for pdf_file_path in initial_docs:
            nodes = generate_vector_store_nodes(pdf_file_path, self.embed_model)
            self.vector_store_index.insert_nodes(nodes)
            # self.vector_store.add(nodes)

        if self.use_vllm:
            # Note: vLLM does not support streaming interface yet
            # ref: https://github.com/run-llama/llama_index/issues/9477
            print(f"Loading LLM from {model_name} using vLLM...")
            from llama_index.llms.vllm import Vllm  # lazy loading

            llm = Vllm(
                model=model_name,
                trust_remote_code=True,  # mandatory for hf models
                max_new_tokens=4096,
                vllm_kwargs=self.vllm_kwargs if self.vllm_kwargs else {},
                top_k=10,
                top_p=0.95,
                temperature=0.8,
            )
        else:
            print(f"Loading LLM from {model_name} using transformers.AutoModelForCausalLM...")
            from llama_index.llms.huggingface import HuggingFaceLLM  # lazy loading

            model_kwargs = {"temperature": 0.8, "do_sample": True, "top_k": 10, "top_p": 0.95}
            if self.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            llm = HuggingFaceLLM(
                model_name=model_name,
                tokenizer_name=model_name,
                max_new_tokens=4096,
                is_chat_model=True,
                model_kwargs=model_kwargs,
            )
            # Overwrite chat_template to support system prompt
            llm._tokenizer.chat_template = CHAT_TEMPLATE

        self.retriever = self.vector_store_index.as_retriever(query_mode="default", similarity_top_k=2)
        self.chat_engine = ContextChatEngine.from_defaults(retriever=self.retriever, llm=llm)

    def add_document(self, list_file_obj: List, progress=gr.Progress()):
        if self.vector_store is None:
            raise ValueError("Vectorstore not initialized. Please run initialize_database() first.")

        if list_file_obj is None:
            return gr.update(value="Upload PDF documents", interactive=True)

        gr.Info(f"Adding {len(list_file_obj)} documents to vector database...")
        # progress(0, desc="Copying documents to ")
        pdf_docs = [x.name for x in list_file_obj if x is not None]
        for pdf in pdf_docs:
            shutil.copy(pdf, self.docs_folder)

        gr.Info("Adding documents into vector database...")
        for pdf_file_path in progress.tqdm(pdf_docs, desc="Copying documents to "):
            nodes = generate_vector_store_nodes(pdf_file_path, self.embed_model)
            self.vector_store_index.insert_nodes(nodes)
            # progress(1, desc=f"Adding {pdf_file_path} to vector database")

        gr.Info("Upload Completed!")
        return gr.update(value="Upload PDF documents", interactive=True)

    def handle_chat(self, message, history):
        if self.stream:
            streaming_response = self.chat_engine.stream_chat(message)
            full_response = ""
            for token in streaming_response.response_gen:
                full_response += token
                yield full_response

            return full_response
        else:
            chat_response = self.chat_engine.chat(message)
            return chat_response.response


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
                initial_docs.append(str(os.path.join(root, file)))

    ragger = RAGInterface(
        embedding_model_name=args.embedding_model_name,
        stream=True if args.stream else False,
        use_flash_attention=True if args.use_flash_attention else False,
        use_vllm=True if args.use_vllm else False,
        vllm_kwargs={
            "max_model_len": int(args.vllm_max_model_len),
            "enforce_eager": args.vllm_enforce_eager,
        } if args.use_vllm else {},

    )
    ragger.initialize_chat_engine(initial_docs, model_name=args.model_name)

    with gr.Blocks(css=css, title="RAG Chatbot with LlamaIndex🦙 and Open-source LLMs") as demo:
        with gr.Row():
            gr.Markdown(
                f"""<h2>RAG Chatbot with LlamaIndex🦙</h2>
                    <h2>Model name: {args.model_name}</h2>
                    <h3>Ask any questions about your PDF documents, along with follow-ups</h3>
                    <b>Note:</b> This AI assistant performs retrieval-augmented generation from your PDF documents.<br>
                    Initial documents are loaded from the `{args.docs_folder}` folder. You can add more documents by 
                    clicking the button below.<br>
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
                        label="Add your PDF documents (single or multiple)"
                    )
                with gr.Row():
                    generate_vectordb_btn = gr.Button("Upload PDF documents")
                    generate_vectordb_btn.click(
                        fn=lambda: gr.update(
                            value="Uploading...",
                            interactive=False
                        ), outputs=[generate_vectordb_btn]
                    ).then(
                        fn=ragger.add_document,
                        inputs=[document],
                        outputs=[generate_vectordb_btn]
                    )
            with gr.Column(scale=2):
                gr.ChatInterface(ragger.handle_chat)
        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
            close_button.click(fn=lambda: gr.update(interactive=False), outputs=[close_button]).then(fn=close_app)

    demo.queue().launch(server_name="0.0.0.0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="RAG Chatbot",
        description="Question Answering with Retrieval QA and LangChain Language Models featuring FAISS vector stores.")

    parser.add_argument("--docs-folder", default="./docs", help="Path to the folder containing the PDF documents.")
    parser.add_argument("--embedding-model-name", default="BAAI/bge-m3",
                        help="HuggingFace model name for text embeddings.")
    parser.add_argument("--model-name", default="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
                        help="HuggingFace model name for LLM.")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for generation.")
    parser.add_argument("--stream", action="store_true", help="Use streaming interface for vLLM generation.")
    parser.add_argument("--use-flash-attention", action="store_true", help="Use Flash Attention for vLLM generation.")
    parser.add_argument("--vllm-max-model-len", default=4096, help="Max model length - Only active if using vLLM")
    parser.add_argument("--vllm-enforce-eager", action="store_true",
                        help="Enforce eager mode - Only active if using vLLM")

    args = parser.parse_args()

    # Validate arguments
    if args.use_vllm:
        print(
            "[WARN] Using vLLM for text generation. You might want to install vLLM with `pip install -U vllm llama-index-llms-vllm` before running the app.")
        if args.stream:
            raise ValueError("vLLM on LlamaIndex does not streaming interface yet. Please remove --stream flag.")
    if args.use_flash_attention:
        print(
            "[WARN] Using Flash Attention for vLLM. You might want to install Flash Attention with `pip install -U flash-attn` before running the app.")

    main(args)
