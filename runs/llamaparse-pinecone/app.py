import argparse
import os
import traceback
from typing import List
import uuid

import gradio as gr
from llama_parse import LlamaParse
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Initialize global variables
pc = None  # Pinecone client
openai_client = None  # OpenAI client
parser = None  # LlamaParse client

# Default prompt
DEFAULT_PROMPT = "You are a helpful AI assistant. Use the following pieces of context to answer the human's question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Context: {context} Conversation history: {history} Human: {message}"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="PDF Parser and RAG Chatbot")
parser.add_argument("--llama_parse_api_key", help="LlamaParse API Key")
parser.add_argument("--pinecone_api_key", help="Pinecone API Key")
parser.add_argument("--openai_api_key", help="OpenAI API Key")
parser.add_argument("--openai_api_base", default="https://api.openai.com/v1", help="OpenAI API Base URL")
parser.add_argument("--pinecone_index_name", default="pdf-parser-index", help="Pinecone Index Name")
parser.add_argument("--pinecone_region", default="us-east-1", help="Pinecone Region")
args = parser.parse_args()

# Set environment variables from command-line arguments
if not args.llama_parse_api_key:
    args.llama_parse_api_key = os.getenv("LLAMA_PARSE_API_KEY")
if not args.pinecone_api_key:
    args.pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not args.openai_api_key:
    args.openai_api_key = os.getenv("OPENAI_API_KEY")

def initialize_pinecone():
    global pc
    pc = Pinecone(api_key=args.pinecone_api_key)

    if args.pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=args.pinecone_index_name,
            dimension=1024,  # Dimensionality of the embeddings (multilingual-e5-large)
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=args.pinecone_region
            )
        )

def initialize_openai():
    global openai_client
    openai_client = OpenAI(
        base_url=args.openai_api_base,
        api_key=args.openai_api_key
    )

def initialize_llama_parse():
    global parser
    parser = LlamaParse(api_key=args.llama_parse_api_key, result_type="text", fast_mode=True)

def get_embedding(text):
    embeddings = pc.inference.embed(
        model="multilingual-e5-large", # https://docs.pinecone.io/guides/inference/understanding-inference#embed
        inputs=[text],
        parameters={"input_type": "passage", "truncate": "END"}
    )
    return embeddings.data[0]["values"]

def chat(message, history, prompt):
    global openai_client, pc
    if openai_client is None or pc is None:
        return "Please initialize the settings first."

    # Get message embedding
    query_embedding = get_embedding(message)

    # Search Pinecone for relevant contexts
    index = pc.Index(args.pinecone_index_name)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    # Prepare context from search results
    contexts = [item.metadata['text'] for item in results['matches']]
    context_str = "\n\n".join(contexts)

    # Prepare conversation history
    conversation = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in history])

    # Use the provided prompt or the default one
    full_prompt = prompt.format(context=context_str, history=conversation, message=message)

    # Generate response using OpenAI
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt}
        ]
    )

    return response.choices[0].message.content

def start_parse():
    return gr.update(interactive=False)

def parse_and_ingest(file_paths: List[str], progress=gr.Progress()):
    global pc, parser
    if pc is None or parser is None:
        gr.Warning("Please initialize the settings first.")
        return [gr.update(interactive=True), gr.update(value="", visible=False)]

    if not file_paths:
        gr.Warning("No files uploaded.")
        return [gr.update(interactive=True), gr.update(value="", visible=False)]

    gr.Info("Parsing documents...")
    index = pc.Index(args.pinecone_index_name)
    progress((0, len(file_paths)), desc=f"Parsing documents")

    try:
        documents = parser.load_data(file_paths)
        progress((len(file_paths), len(file_paths)+len(documents)), desc=f"Ingesting documents into vector database")

        for i, doc in enumerate(documents):
            progress((len(file_paths)+i, len(file_paths)+len(documents)))

            embedding = get_embedding(doc.text)
            index.upsert(vectors=[
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {"text": doc.text}
                }
            ])

        gr.Info("Parsed and ingested {processed_chunks} chunks from {len(file_paths)} documents into the vector database.")
        return [gr.update(interactive=True), gr.update(value="", visible=False)]

    except Exception as e:
        gr.Warning(f"Upload and ingestion failed: {str(e)}")
        return [gr.update(interactive=True), gr.update(value=traceback.format_exc(), visible=True)]

def browse_vectors(query):
    if pc is None:
        return "Please initialize the settings first."

    query_embedding = get_embedding(query)
    index = pc.Index(args.pinecone_index_name)
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    return "\n\n".join([f"Result {i+1}:\n{item.metadata['text']}" for i, item in enumerate(results['matches'])])

def remove_vector(query):
    if pc is None:
        return "Please initialize the settings first."

    query_embedding = get_embedding(query)
    index = pc.Index(args.pinecone_index_name)
    results = index.query(vector=query_embedding, top_k=1, include_metadata=True)

    if results['matches']:
        index.delete(ids=[results['matches'][0].id])
        return f"Removed vector with ID: {results['matches'][0].id}"
    else:
        return "No matching vector found to remove."

def start_update():
    gr.Info("Updating settings...")
    return gr.update(interactive=False)

def update_settings(openai_api_key, llama_parse_api_key, pinecone_api_key, pinecone_region, openai_api_base):
    global args
    args.openai_api_key = openai_api_key
    args.llama_parse_api_key = llama_parse_api_key
    args.pinecone_api_key = pinecone_api_key
    args.pinecone_region = pinecone_region
    args.openai_api_base = openai_api_base

    try:
        initialize_pinecone()
        initialize_openai()
        initialize_llama_parse()
        gr.Info("Settings updated and initialized successfully!")
        return [gr.update(interactive=True), gr.update(value="", visible=False)]
    except Exception:
        gr.Info(f"Error updating settings - refer to error message for details.")
        return [gr.update(interactive=True), gr.update(value=traceback.format_exc(), visible=True)]

# Gradio interface
with gr.Blocks(title="🦙 PDF RAG demo with LlamaIndex and Pinecone") as demo:
    gr.Markdown("# 🦙 PDF RAG demo with LlamaIndex and Pinecone!")

    gr.Markdown("""
    Parse PDF documents and chat with an AI about the content using LlamaParse, Pinecone, and self-hosted LLMs!

    * Instructions:
      1. **Chat Tab**: Ask questions about the ingested documents and receive AI-generated responses.
      2. **Document Tab**: Upload and ingest PDF documents into the vector database using LlamaParse.
      3. **Vector Browser Tab**: Search for specific information in the Pinecone vector database or remove entries.
      4. **Settings Tab**: Configure your API keys and system prompts
    """)

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Chat Message")
        clear = gr.Button("Clear")

    with gr.Tab("Document"):
        file_input = gr.File(label="Upload PDF", file_count="multiple")
        parse_button = gr.Button("Parse and Ingest")
        parse_error_msg = gr.Textbox(label="Error Message", visible=False, value="", lines=10)

    with gr.Tab("Vector Browser"):
        search_input = gr.Textbox(label="Search Query")
        search_button = gr.Button("Search")
        remove_button = gr.Button("Remove")
        vector_output = gr.Textbox(label="Results")

    with gr.Tab("Settings"):
        with gr.Group():
            gr.Markdown("### 🌲 Pinecone Settings")
            pinecone_api_key = gr.Textbox(label="Pinecone API Key", value=args.pinecone_api_key)
            pinecone_region = gr.Textbox(label="Pinecone Region", value=args.pinecone_region)
            pinecone_index_name = gr.Textbox(label="Pinecone Index Name", value=args.pinecone_index_name)

        with gr.Group():
            gr.Markdown("### 🦙 LlamaCloud Settings")
            llama_parse_api_key = gr.Textbox(label="LlamaCloud API Key", value=args.llama_parse_api_key)

        with gr.Group():
            gr.Markdown("### 🧠 LLM Settings")
            openai_api_key = gr.Textbox(label="OpenAI API Key", value=args.openai_api_key)
            openai_api_base = gr.Textbox(label="OpenAI API Base URL", value=args.openai_api_base)
            prompt_input = gr.Textbox(label="Prompt Template", value=DEFAULT_PROMPT, lines=5)

        update_settings_button = gr.Button("Update")
        update_settings_error_msg = gr.Textbox(label="Error Message", visible=False, value="", lines=10)

    msg.submit(chat, inputs=[msg, chatbot, prompt_input], outputs=[chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    parse_button.click(
        start_parse, outputs=[parse_button]
    ).then(parse_and_ingest, inputs=[file_input], outputs=[parse_button, parse_error_msg])
    search_button.click(browse_vectors, inputs=[search_input], outputs=[vector_output])
    remove_button.click(remove_vector, inputs=[search_input], outputs=[vector_output])
    update_settings_button.click(
        start_update, outputs=[update_settings_button]
    ).then(
        update_settings,
        inputs=[openai_api_key, llama_parse_api_key, pinecone_api_key, pinecone_region, openai_api_base],
        outputs=[update_settings_button, update_settings_error_msg]
    )

if __name__ == "__main__":
    initialize_pinecone()
    initialize_openai()
    initialize_llama_parse()
    demo.queue().launch(server_name="0.0.0.0")
