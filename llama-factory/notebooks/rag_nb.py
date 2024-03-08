# %%
import torch
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig, MixtralForCausalLM


# %%
DOCS_DIR = "/vessl-docs"

LORA_PATH = "/lora/checkpoint-100"
QLORA_PATH = "/qlora/checkpoint-100"

# %%
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# %%
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = MixtralForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config,
)

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

# %%
documents = SimpleDirectoryReader(DOCS_DIR, recursive=True).load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# %%
question = "How can I create a new model via web?"

# %%
base_llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=512,
    model=model,
    tokenizer=tokenizer,
)

query_engine = index.as_query_engine(llm=base_llm)

response = query_engine.query(question)
print("==BASE MODEL==")
print(response.response)

# %%
lora_model = PeftModel.from_pretrained(model, LORA_PATH)
lora_llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=512,
    model=lora_model,
    tokenizer=tokenizer,
)

query_engine = index.as_query_engine(llm=lora_llm)

response = query_engine.query(question)
print("==LORA MODEL==")
print(response.response)

# %%
qlora_model = PeftModel.from_pretrained(model, QLORA_PATH)
qlora_llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=512,
    model=qlora_model,
    tokenizer=tokenizer,
)

query_engine = index.as_query_engine(llm=base_llm)

response = query_engine.query(question)
print("==QLORA MODEL==")
print(response.response)

# %%
