import os
import pickle

import chromadb
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings


chroma_path = os.environ.get("CHROMA_PATH", "/chroma")
collection_name = os.environ.get("CHROMA_COLLECTION", "fever-wiki-pages")
embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
data_path = os.environ.get("DATA_PATH", "/data")


# initialize Chroma
print(f"initializing Chroma from {chroma_path}...")
client = chromadb.PersistentClient(chroma_path)
_collection = client.get_or_create_collection(collection_name)
vector_store = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=HuggingFaceBgeEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda"},
    ),
)

# load wiki pages
print("loading wiki pages...")
data = load_dataset("fever", "wiki_pages", trust_remote_code=True)
ids = []
texts = []
n = 0
for page in data["wikipedia_pages"]:
    if page["text"]:
        ids.append(page["id"])
        texts.append(page["text"])
        n += 1
    if n == 200000:
        break

assert len(ids) == len(texts)

# add wiki pages to chroma
docs = [Document(id=id, page_content=text) for id, text in zip(ids, texts)]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
text_chunks = text_splitter.split_documents(docs)
print(f"Adding {len(text_chunks)} documents to Chroma...")

batch_size = 40000
for i in range(0, len(text_chunks), batch_size):
    print(f"Adding documents #{i} ~ #{i + batch_size}...")
    vector_store.add_documents(text_chunks[i:i + batch_size])
    print(f"Added documents #{i} ~ #{i + batch_size}.")

# load claims
print("loading claims...")
data = load_dataset("fever", "v1.0", trust_remote_code=True)
claims = [page for page in data["train"] if page["evidence_wiki_url"] in ids]
print(f"claims length: {len(claims)}")

claims_path = os.path.join(data_path, "claims.pkl")
with open(claims_path, "wb") as fw:
    pickle.dump(claims, fw)
print(f"claims saved to {claims_path}")
