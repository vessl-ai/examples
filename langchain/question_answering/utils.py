import argparse
import os
import logging

from langchain.llms import OpenAI
from langchain.document_loaders import *
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import *
from langchain.chains import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_url", type=str, default=None)
    parser.add_argument("--file_dir", type=str, default=None)
    parser.add_argument("--files", type=str, nargs="+", default=None)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    parser.add_argument("--embedding_type", type=str, default="openai")
    parser.add_argument("--llm_type", type=str, default="openai")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=-1)
    parser.add_argument("--branch", type=str, default="main")
    return parser.parse_args()


def get_embeddings(embedding_type="openai"):
    if embedding_type == "openai":
        assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is None"
        return OpenAIEmbeddings(allowed_special={'<|endoftext|>'})
    elif embedding_type == "AlephAlpha":
        return AlephAlphaAsymmetricSemanticEmbedding()
    elif embedding_type == "Cohere":n
        assert os.getenv("COHERE_API_KEY") is not None, "COHERE_API_KEY is None"
        return CohereEmbeddings()


def get_llm(llm_type="openai", temperature=0.9, n=1, max_tokens=-1):
    if llm_type == "openai":
        assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is None"
        return OpenAI(temperature=temperature, n=n, max_tokens=max_tokens)


def set_api_keys(keys):
    if keys is not None:
        for key in keys:
            os.environ[key[0]] = key[1]


def make_qa(repo_url=None, branch="main", files=None, file_dir=None, chunk_size=500, chunk_overlap=100,
            embedding_type="openai"):
    embeddings = get_embeddings(embedding_type=embedding_type)
    documents = []
    if repo_url is not None:
        if os.path.exists(repo_url.split('/')[-1]):
            os.system(f"rm -rf {repo_url.split('/')[-1]}")
        try:
            documents.extend(
                GitLoader(clone_url=repo_url, repo_path=f"./{repo_url.split('/')[-1]}", branch=branch).load())
        except:
            # log error
            logging.log(logging.WARNING, f"Failed in cloning repo: {repo_url}")
            pass

    if file_dir is not None:
        documents.extend(DirectoryLoader(directory_path=file_dir).load())

    if files is not None:
        for file in files:
            format = file.split(".")[-1]
            if format == "pdf":
                documents.extend(PyPDFLoader(file_path=file).load())
            elif format == "py":
                documents.extend(PythonLoader(file_path=file).load())
            elif format == "csv":
                documents.extend(CSVLoader(file_path=file).load())
            else:
                documents.extend(TextLoader(file_path=file).load())

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    docsearch = Chroma.from_documents(texts, embeddings)
    # make chain
    qa = ConversationalRetrievalChain.from_llm(llm=OpenAI(temperature=0.7),
                                               retriever=docsearch.as_retriever()
                                               )
    return qa
