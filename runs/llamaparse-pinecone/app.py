import argparse
import os
import traceback
from typing import List
import uuid

import gradio as gr
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai_like import OpenAILike
from llama_parse import LlamaParse
from pinecone import Pinecone, ServerlessSpec

# Initialize global variables
pc = None  # Pinecone client
openailike_client = None  # OpenAILike client
parser = None  # LlamaParse client
system_prompt_template = "You are a helpful AI assistant. Use the following pieces of context to answer the human's question. If you don't know the answer, just say that you can't find the answer from the context, don't try to make up an answer. Context: {context}"
uploaded_files = []

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Chat-with-document demo")
parser.add_argument("--llama-parse-api-key", required=False, help="LlamaCloud API Key")
parser.add_argument("--pinecone-api-key", required=False, help="Pinecone API Key")
parser.add_argument("--openai-api-base", default="https://api.openai.com/v1", help="Base URL of OpenAI-compatible API")
parser.add_argument("--openai-api-key", required=False, help="API Key for OpenAI-compatible API")
parser.add_argument("--openai-api-model", default="gpt-4o-mini", help="Model served by OpenAI-compatible API")
parser.add_argument("--pinecone-index-name", default="pdf-parser-index", help="Pinecone Index Name")
parser.add_argument("--pinecone-region", default="us-east-1", help="Pinecone Region")
args = parser.parse_args()

# Set environment variables from command-line arguments
if not args.llama_parse_api_key:
    args.llama_parse_api_key = os.getenv("LLAMA_PARSE_API_KEY")
if not args.pinecone_api_key:
    args.pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not args.openai_api_key:
    args.openai_api_key = os.getenv("OPENAI_API_KEY")


def initialize_pinecone():
    if not args.pinecone_api_key:
        raise ValueError("Pinecone API key is not provided.")

    global pc
    pc = Pinecone(api_key=args.pinecone_api_key)

    try:
        indices = pc.list_indexes()
        if args.pinecone_index_name not in indices.names():
            pc.create_index(
                name=args.pinecone_index_name,
                dimension=1024,  # Dimensionality of the embeddings (multilingual-e5-large)
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=args.pinecone_region
                )
            )
    except Exception as e:
        pc = None
        raise e


def initialize_openai():
    global openailike_client
    print(args.openai_api_base, args.openai_api_model, args.openai_api_key)
    openailike_client = OpenAILike(
        model=args.openai_api_model,
        is_chat_model=True,
        api_base=args.openai_api_base,
        api_key=args.openai_api_key if args.openai_api_key else "tgi")


def initialize_llama_parse():
    if not args.llama_parse_api_key:
        raise ValueError("LlamaCloud API key is not provided.")

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
    if openailike_client is None or pc is None or parser is None:
        yield "💡 Please initialize the settings first on the 'Settings' tab."
        return

    try:
        # Get message embedding
        query_embedding = get_embedding(message)

        # Search Pinecone for relevant contexts
        index = pc.Index(args.pinecone_index_name)
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    except Exception as e:
        yield f"💡 Error querying Pinecone, Please update the prompt or reinitialize Pinecone on the 'Settings' tab.\nError: {str(e)}"
        return

    # Prepare context from search results
    contexts = [item.metadata['text'] for item in results['matches']]
    context_str = "\n\n".join(contexts)

    # Use the provided prompt template
    system_prompt = system_prompt_template.format(context=context_str)

    # Generate response using OpenAI
    response = openailike_client.stream_chat(
        messages=[
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=message),
        ],
    )

    for chunk in response:
        yield str(chunk.message.content)

def start_parse():
    return gr.update(value="Parsing documents...", interactive=False)

def parse_and_ingest(new_files: List[str], progress=gr.Progress()):
    global pc, parser, uploaded_files
    if pc is None or parser is None:
        gr.Warning("💡 Please initialize the settings first on the 'Settings' tab.")
        return [
            gr.update(value=[]),
            gr.update(value="Parse and Ingest", interactive=True),
            gr.update(value="", visible=False),
            gr.update(value=uploaded_files)
        ]

    if not new_files:
        gr.Warning("No new files uploaded.")
        return [
            gr.update(value=[]),
            gr.update(value="Parse and Ingest", interactive=True),
            gr.update(value="", visible=False),
            gr.update(value=uploaded_files)
        ]

    gr.Info("Parsing documents...")
    index = pc.Index(args.pinecone_index_name)
    progress((0, len(new_files)), desc=f"Upload & parse documents using LlamaParse")

    try:
        documents = parser.load_data(new_files)
        progress((len(new_files), len(new_files)+len(documents)), desc=f"Ingesting documents into vector database")

        for i, doc in enumerate(documents):
            progress((len(new_files)+i, len(new_files)+len(documents)), desc=f"Ingesting documents into vector database")

            embedding = get_embedding(doc.text)
            index.upsert(vectors=[
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {"text": doc.text}
                }
            ])

        # Add successfully processed files to the uploaded_files list
        uploaded_files.extend(new_files)

        gr.Info(f"Parsed and ingested {len(documents)} documents from {len(new_files)} files into the vector database.")
        return [
            gr.update(value=[]),
            gr.update(value="Parse and Ingest", interactive=True),
            gr.update(value="", visible=False),
            gr.update(value=uploaded_files)
        ]

    except Exception as e:
        gr.Warning(f"Upload and ingestion failed: {str(e)}")
        return [
            gr.update(value=[]),
            gr.update(value="Parse and Ingest", interactive=True),
            gr.update(value=traceback.format_exc(), visible=True),
            gr.update(value=uploaded_files)  # Return the existing list of uploaded files
        ]

def clear_uploaded_files():
    global uploaded_files
    uploaded_files = []
    return gr.update(value=[])

def browse_vectors(query):
    if pc is None:
        return [[], gr.update(value="💡 Please initialize the settings first on the 'Settings' tab.", visible=True)]

    try:
        query_embedding = get_embedding(query)
        index = pc.Index(args.pinecone_index_name)
        results = index.query(vector=query_embedding, top_k=10, include_metadata=True)

        vectors_data = []
        for item in results['matches']:
            vectors_data.append([
                item.score,
                item.metadata['text'][:500] + "...",  # Truncate text to 500 characters
                item.id
            ])

        return [vectors_data, gr.update(value="", visible=False)]
    except Exception as e:
        gr.Warning(f"Upload and ingestion failed: {str(e)}")
        return [[], gr.update(value=traceback.format_exc(), visible=True)]

def start_update():
    gr.Info("Updating settings...")
    return gr.update(interactive=False)

def update_settings(llama_parse_api_key, pinecone_api_key, pinecone_region, pinecone_index_name, openai_api_base, openai_api_model, openai_api_key, prompt_input):
    global args, system_prompt_template
    args.llama_parse_api_key = llama_parse_api_key
    args.pinecone_api_key = pinecone_api_key
    args.pinecone_region = pinecone_region
    args.pinecone_index_name = pinecone_index_name
    args.openai_api_base = openai_api_base
    args.openai_api_model = openai_api_model
    args.openai_api_key = openai_api_key
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
.gradio-container { height: 170vh !important; }
#component-0, .tabs { height: 100%; }

.tabs, #chat-container.tabitem .gap { height: 100%; }
#chat-container.tabitem { height: 48%; }
"""

# Gradio interface
with gr.Blocks(css=css, fill_height=True, title="🦙 Chat-with-document demo with LlamaIndex and Pinecone") as demo:
    gr.Markdown("# 🦙 Chat-with-document Demo with LlamaIndex and Pinecone!")

    gr.Markdown("""
    Parse PDF documents and chat with an AI about the content using LlamaParse, Pinecone, and self-hosted LLMs!

    * Instructions:
      1. **Chat Tab**: Ask questions about the ingested documents and receive AI-generated responses.
      2. **Document Tab**: Upload and ingest PDF documents into the vector database using LlamaParse.
      3. **Vector Browser Tab**: Search for specific information in the Pinecone vector database or remove entries.
      4. **Settings Tab**: Configure your API keys and system prompts.
    """)

    with gr.Tab("Chat", elem_id="chat-container"):
        gr.ChatInterface(handle_chat)

    with gr.Tab("Document"):
        with gr.Row():
            with gr.Column(scale=1):
                uploaded_files_view = gr.Files(label="Uploaded Files", interactive=False, value=uploaded_files)
            with gr.Column(scale=3):
                file_input = gr.File(label="Upload PDF", file_count="multiple")
        parse_button = gr.Button("Parse and Ingest")
        parse_error_msg = gr.Textbox(label="Error Message", visible=False, value="", lines=10)

    with gr.Tab("Vector Browser"):
        with gr.Row():
            search_input = gr.Textbox(label="Search Query", scale=7)
            search_button = gr.Button("Search", scale=1)

        vector_output = gr.DataFrame(
            headers=["score", "text", "id"],
            label="Search Results",
            interactive=False
        )
        vector_search_error_msg = gr.Textbox(label="Error Message", visible=False, value="", lines=10)

    with gr.Tab("Settings", elem_id="settings-container"):
        with gr.Group():
            gr.Markdown("### 🦙 LlamaCloud Settings\n\t* Obtain an API key from the [LlamaCloud dashboard]( ttps://cloud.llamaindex.ai/api-key).")
            llama_parse_api_key = gr.Textbox(label="LlamaCloud API Key", value=args.llama_parse_api_key)

        with gr.Group():
            gr.Markdown("### 🌲 Pinecone Settings\n\t* Retrieve an API key from the [Pinecone console](https://app.pinecone.io/organizations/-/projects/-/keys).")
            gr.Markdown
            pinecone_api_key = gr.Textbox(label="Pinecone API Key", value=args.pinecone_api_key)
            pinecone_region = gr.Textbox(label="Pinecone Region", value=args.pinecone_region)
            pinecone_index_name = gr.Textbox(label="Pinecone Index Name", value=args.pinecone_index_name)

        with gr.Group():
            gr.Markdown("### 🧠 LLM Settings\n\t* For OpenAI: Obtain an API key from [OpenAI's API Key page](https://platform.openai.com/api-keys). For OpenAI-compatible endpoints (e.g., vLLM, TGI), update the API Base URL and key accordingly.")
            openai_api_base = gr.Textbox(label="OpenAI-compatible API Base URL", value=args.openai_api_base)
            openai_api_model = gr.Textbox(label="LLMs to use", value=args.openai_api_model)
            openai_api_key = gr.Textbox(label="OpenAI-compatible API Key", value=args.openai_api_key)
            prompt_input = gr.Textbox(label="Prompt Template", value=system_prompt_template, lines=5)

        update_settings_button = gr.Button("Update")
        update_settings_error_msg = gr.Textbox(label="Error Message", visible=False, value="", lines=10)

    parse_button.click(
        start_parse, outputs=[parse_button]
    ).then(
        parse_and_ingest,
        inputs=[file_input],
        outputs=[file_input, parse_button, parse_error_msg, uploaded_files_view]
    )
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
        inputs=[llama_parse_api_key, pinecone_api_key, pinecone_region, pinecone_index_name, openai_api_base, openai_api_model, openai_api_key, prompt_input],
        outputs=[update_settings_button, update_settings_error_msg]
    )

if __name__ == "__main__":
    try:
        initialize_pinecone()
    except Exception as e:
        print(f"Error initializing Pinecone, skipping initialization: {str(e)}")
    try:
        initialize_openai()
    except Exception as e:
        print(f"Error initializing OpenAI, skipping initialization: {str(e)}")
    try:
        initialize_llama_parse()
    except Exception as e:
        print(f"Error initializing LlamaParse, skipping initialization: {str(e)}")
    demo.queue().launch()
