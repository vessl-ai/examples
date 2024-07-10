import os
import pickle

import chromadb
from datasets import Dataset
from langchain_chroma import Chroma
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chroma_path = os.environ.get("CHROMA_PATH", "/chroma")
collection_name = os.environ.get("CHROMA_COLLECTION", "fever-wiki-pages")
embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
data_path = os.environ.get("DATA_PATH", "/data")
llm_endpoint = os.environ.get("LLM_ENDPOINT", "https://api.openai.com")
llm_model = os.environ.get("LLM_MODEL", "gpt-4o")
rag_pattern = os.environ.get("RAG_PATTERN", "naive")
reranker_model = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
output_path = os.environ.get("OUTPUT_PATH", "/output")


# initialize Chroma
print(f"initializing Chroma from {chroma_path}...")
client = chromadb.PersistentClient(chroma_path)
_collection = client.get_collection(collection_name)
vector_store = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={
            "device": "cuda:0",
            "trust_remote_code": True,
        },
    ),
)

# load claims
print("loading claims...")
claims_path = os.path.join(data_path, "claims.pkl")
with open(claims_path, "rb") as fr:
    claims = pickle.load(fr)
print(f"claims length: {len(claims)}")

# initialize RAG chain
prompt = PromptTemplate.from_template(
    "As an AI assistant, you are assigned to perform verification tasks. Utilize the provided retrieved context to formulate your answer whether the given claim is true or false. If the context doesn't contain sufficient information for a direct or comprehensive response, clarify that the given context didn't offer enough information to provide a fully accurate response.\n"
    "Presented Claim: {question}\n"
    "Provided Context: {context}"
)

retriever = vector_store.as_retriever()

print("LLM endpoint:", llm_endpoint)
base_url = os.path.join(llm_endpoint, "v1")
llm = ChatOpenAI(
    base_url=base_url,
    model=llm_model,
    temperature=0,
    max_tokens=4096,
    streaming=True,
)

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG pattern:", rag_pattern)
match rag_pattern:
    case "naive":
        retrieve_docs = (lambda x: x["question"]) | retriever

    case "hyde":
        hyde_prompt = PromptTemplate.from_template(
            "Please write a passage to tell whether the given claim is true or not.\n"
            "Claim: {question}\n"
            "Passage:"
        )

        hyde_llm = ChatOpenAI(
            base_url=base_url,
            model=llm_model,
            temperature=0,
            max_tokens=4096,
            streaming=True,
        )

        hyde_chain = (
            {"question": RunnablePassthrough()}
            | hyde_prompt
            | hyde_llm
            | StrOutputParser()
        )

        retrieve_docs = (lambda x: x["question"]) | hyde_chain | retriever

    case "reranking":
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5},
        )

        encoder = HuggingFaceCrossEncoder(
            model_name=reranker_model,
            model_kwargs={
                "device": "cuda:1",
                "trust_remote_code": True,
            },
        )
        compressor = CrossEncoderReranker(model=encoder, top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        retrieve_docs = (lambda x: x["question"]) | compression_retriever

    case "hyde-reranking":
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5},
        )

        encoder = HuggingFaceCrossEncoder(
            model_name=reranker_model,
            model_kwargs={
                "device": "cuda:1",
                "trust_remote_code": True,
            },
        )
        compressor = CrossEncoderReranker(model=encoder, top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        hyde_prompt = PromptTemplate.from_template(
            "Please write a passage to tell whether the given claim is true or not.\n"
            "Claim: {question}\n"
            "Passage:"
        )

        hyde_llm = ChatOpenAI(
            base_url=base_url,
            model=llm_model,
            temperature=0,
            max_tokens=4096,
            streaming=True,
        )

        hyde_chain = (
            {"question": RunnablePassthrough()}
            | hyde_prompt
            | hyde_llm
            | StrOutputParser()
        )

        retrieve_docs = (lambda x: x["question"]) | hyde_chain | compression_retriever
    
    case _:
        raise ValueError(f"Invalid RAG pattern: {rag_pattern}")

rag_chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
    answer=rag_chain_from_docs
)

# get RAG results
print("Getting RAG results...")
inputs = [{"question": c["claim"]} for c in claims]

if rag_pattern in ["naive", "hyde"]:
    rag_results = rag_chain.batch(inputs)
else:  # rag_chain.batch() raises an error for reranking patterns
    rag_results = [
        rag_chain.invoke(ip) for ip in inputs
    ]

# save RAG results
dataset = Dataset.from_dict({
    "question": [r["question"] for r in rag_results],
    "contexts": [[d.page_content for d in r["context"]] for r in rag_results],
    "answer": [r["answer"] for r in rag_results],
    "ground_truth": ["TRUE" if c["label"] == "SUPPORTS" else "FALSE" for c in claims]
})

dataset_path = os.path.join(output_path, f"dataset-{rag_pattern}")
os.makedirs(output_path, exist_ok=True)
dataset.save_to_disk(dataset_path)
print(f"Saved RAG results to {dataset_path}.")
