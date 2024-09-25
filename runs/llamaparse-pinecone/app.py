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
system_prompt_template = "You are a helpful AI assistant. Use the following pieces of context to answer the human's question. If you don't know the answer, just say that you can't find the answer from the context, don't try to make up an answer. Context: {context}"

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

def handle_chat(message, history):
    if openai_client is None or pc is None:
        yield "💡 Please initialize the settings first on the 'Settings' tab."
        return

    # Prepare conversation history
    print(history)
    conversation = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in history])

    # Get message embedding
    query_embedding = get_embedding(message)

    # Search Pinecone for relevant contexts
    index = pc.Index(args.pinecone_index_name)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    # Prepare context from search results
    contexts = [item.metadata['text'] for item in results['matches']]
    context_str = "\n\n".join(contexts)

    # Use the provided prompt template
    system_prompt = system_prompt_template.format(context=context_str)

    # Generate response using OpenAI
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        stream=True
    )

    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            yield full_response

def start_parse():
    return gr.update(value="Parsing documents...", interactive=False)

def parse_and_ingest(file_paths: List[str], progress=gr.Progress()):
    global pc, parser
    if pc is None or parser is None:
        gr.Warning("💡 Please initialize the settings first on the 'Settings' tab.")
        return [
            gr.update(value=file_paths),
            gr.update(value="Parse and Ingest", interactive=True),
            gr.update(value="", visible=False)
        ]

    if not file_paths:
        gr.Warning("No files uploaded.")
        return [
            gr.update(value=[]),
            gr.update(value="Parse and Ingest", interactive=True),
            gr.update(value="", visible=False)
        ]

    gr.Info("Parsing documents...")
    index = pc.Index(args.pinecone_index_name)
    progress((0, len(file_paths)), desc=f"Parsing documents")

    try:
        documents = parser.load_data(file_paths)
        progress((len(file_paths), len(file_paths)+len(documents)), desc=f"Ingesting documents into vector database")

        for i, doc in enumerate(documents):
            progress((len(file_paths)+i, len(file_paths)+len(documents)), desc=f"Ingesting documents into vector database")

            embedding = get_embedding(doc.text)
            index.upsert(vectors=[
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {"text": doc.text}
                }
            ])

        gr.Info(f"Parsed and ingested {len(documents)} documents from {len(file_paths)} files into the vector database.")
        return [
            gr.update(value=[]),
            gr.update(value="Parse and Ingest", interactive=True),
            gr.update(value="", visible=False)
        ]

    except Exception as e:
        gr.Warning(f"Upload and ingestion failed: {str(e)}")
        return [
            gr.update(value=[]),
            gr.update(value="Parse and Ingest", interactive=True),
            gr.update(value=traceback.format_exc(), visible=True)
        ]

def browse_vectors(query):
    if pc is None:
        return [[], gr.update(value="💡 Please initialize the settings first on the 'Settings' tab.", visible=True)]

    try:
        query_embedding = get_embedding(query)
        index = pc.Index(args.pinecone_index_name)
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

        vectors_data = []
        for item in results['matches']:
            vectors_data.append([
                item.id,
                item.score,
                item.metadata['text'][:1000] + "..."  # Truncate text to 1000 characters
            ])

        return [vectors_data, gr.update(value="", visible=False)]
    except Exception as e:
        gr.Warning(f"Upload and ingestion failed: {str(e)}")
        return [[], gr.update(value=traceback.format_exc(), visible=True)]

def start_update():
    gr.Info("Updating settings...")
    return gr.update(interactive=False)

def update_settings(openai_api_key, llama_parse_api_key, pinecone_api_key, pinecone_region, openai_api_base, prompt_input):
    global args, system_prompt_template
    args.openai_api_key = openai_api_key
    args.llama_parse_api_key = llama_parse_api_key
    args.pinecone_api_key = pinecone_api_key
    args.pinecone_region = pinecone_region
    args.openai_api_base = openai_api_base
    system_prompt_template = prompt_input

    try:
        initialize_pinecone()
        initialize_openai()
        initialize_llama_parse()
        gr.Info("Settings updated and initialized successfully!")
        return [gr.update(interactive=True), gr.update(value="", visible=False)]
    except Exception:
        gr.Info(f"Error updating settings - refer to error message for details.")
        return [gr.update(interactive=True), gr.update(value=traceback.format_exc(), visible=True)]

css = """
footer {visibility: hidden}
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 150vh !important; }
#component-0, .tabs { height: 100%; }

.tabs, #chat-container.tabitem .gap { height: 100%; }
#chat-container.tabitem { height: 55%; }
"""

# Gradio interface
with gr.Blocks(css=css, fill_height=True, title="🦙 PDF RAG demo with LlamaIndex and Pinecone") as demo:
    gr.Markdown("# 🦙 PDF RAG demo with LlamaIndex and Pinecone!")

    gr.Markdown("""
    Parse PDF documents and chat with an AI about the content using LlamaParse, Pinecone, and self-hosted LLMs!

    * Instructions:
      1. **Chat Tab**: Ask questions about the ingested documents and receive AI-generated responses.
      2. **Document Tab**: Upload and ingest PDF documents into the vector database using LlamaParse.
      3. **Vector Browser Tab**: Search for specific information in the Pinecone vector database or remove entries.
      4. **Settings Tab**: Configure your API keys and system prompts
    """)

    with gr.Tab("Chat", elem_id="chat-container"):
        gr.ChatInterface(handle_chat)

    with gr.Tab("Document"):
        file_input = gr.File(label="Upload PDF", file_count="multiple")
        parse_button = gr.Button("Parse and Ingest")
        parse_error_msg = gr.Textbox(label="Error Message", visible=False, value="", lines=10)

    with gr.Tab("Vector Browser"):
        with gr.Row():
            search_input = gr.Textbox(label="Search Query", scale=7)
            search_button = gr.Button("Search", scale=1)

        vector_output = gr.DataFrame(
            headers=["id", "score", "text"],
            label="Search Results",
            interactive=False
        )
        vector_search_error_msg = gr.Textbox(label="Error Message", visible=False, value="", lines=10)

    with gr.Tab("Settings", elem_id="settings-container"):
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
            prompt_input = gr.Textbox(label="Prompt Template", value=system_prompt_template, lines=5)

        update_settings_button = gr.Button("Update")
        update_settings_error_msg = gr.Textbox(label="Error Message", visible=False, value="", lines=10)

    parse_button.click(
        start_parse, outputs=[parse_button]
    ).then(parse_and_ingest, inputs=[file_input], outputs=[file_input, parse_button, parse_error_msg])
    search_input.submit(
        browse_vectors,
        inputs=[search_input],
        outputs=[vector_output, vector_search_error_msg]
    )
    search_button.click(
        browse_vectors,
        inputs=[search_input],
        outputs=[vector_output, vector_search_error_msg]
    )
    update_settings_button.click(
        start_update, outputs=[update_settings_button]
    ).then(
        update_settings,
        inputs=[openai_api_key, llama_parse_api_key, pinecone_api_key, pinecone_region, openai_api_base, prompt_input],
        outputs=[update_settings_button, update_settings_error_msg]
    )

if __name__ == "__main__":
    initialize_pinecone()
    initialize_openai()
    initialize_llama_parse()
    demo.queue().launch(server_name="0.0.0.0")
