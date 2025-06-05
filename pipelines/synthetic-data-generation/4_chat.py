import argparse
import os
from time import sleep
from typing import List

import chromadb
import gradio as gr
import torch

from langchain_chroma import Chroma
from langchain_community.llms import VLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class LLMChatHandler:
    def __init__(
        self,
        embedding_model: str,
        llm_model: str,
        chroma_path: str,
        collection_name: str,
    ) -> None:
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        self.chroma_client = chromadb.PersistentClient(chroma_path)

        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
        )
        self.retriever = self.vector_store.as_retriever()

        self.llm = VLLM(
            model=llm_model,
            vllm_kwargs={
                "max_model_len": 16384,
            },
        )

        self.prompt = ChatPromptTemplate([
            ("system", "You are a support engineer in VESSL AI. Your task is to answer the question from the client and help them better understand VESSL AI and its solutions(VESSL Run, VESSL Services and VESSL Pipeline) better. Answer the <question> based on the <context> given. If there is no information in the <context>, specify that you cannot answer the question."),
            ("system", "<context> {context} </context>"),
            ("user", "<question> {question} </question>"),
        ])

        self.chain_from_docs = RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"]))) | self.prompt | self.llm | StrOutputParser()
        self.retrieve_docs = (lambda x: x["question"]) | self.retriever
        self.rag_chain = RunnablePassthrough.assign(context=self.retrieve_docs).assign(answer=self.chain_from_docs)

    def rag_query(self, message, history):
        return self.rag_chain.invoke({"question": message})["answer"]
        
    def close_app(self):
        gr.Info("Terminated the app!")
        sleep(1)
        os._exit(0)


def main(args):
    handler = LLMChatHandler(
        args.embedding_model,
        args.llm_model,
        args.chroma_path,
        args.collection_name,
    )

    with gr.Blocks(
        title=f"❄️ Cortex Chatbot with {args.llm_model}", fill_height=True
    ) as demo:
        with gr.Row():
            gr.Markdown(
                f"<h2>Chatbot with {args.llm_model}</h2>"
                "<h3>Interact with LLM using chat interface!<br></h3>"
            )
        gr.ChatInterface(handler.rag_query)
        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
            close_button.click(
                fn=lambda: gr.update(interactive=False), outputs=[close_button]
            ).then(fn=handler.close_app)

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with LLM using Chroma")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3", help="Name of the embedding to use")
    parser.add_argument("--llm-model", type=str, default="google/gemma-3-12b-it", help="name of the LLM model to use")
    parser.add_argument("--chroma-path", type=str, default="/chroma", help="Path to the Chroma database")
    parser.add_argument("--collection-name", type=str, default="synthetic_data", help="Name of the Chroma collection")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    args = parser.parse_args()

    main(args)
