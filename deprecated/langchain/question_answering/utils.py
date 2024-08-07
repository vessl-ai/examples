import logging
import os

from langchain.chains import *
from langchain.document_loaders import *
from langchain.embeddings import *
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def get_embeddings(embedding_type="openai"):
    if embedding_type == "openai":
        assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is None"
        return OpenAIEmbeddings(allowed_special={"<|endoftext|>"})
    elif embedding_type == "AlephAlpha":
        return AlephAlphaAsymmetricSemanticEmbedding()
    elif embedding_type == "Cohere":
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


def make_qa(
    repo_url=None,
    branch="main",
    files=None,
    file_dir=None,
    chunk_size=500,
    chunk_overlap=100,
    embedding_type="openai",
):
    embeddings = get_embeddings(embedding_type=embedding_type)
    documents = []
    if repo_url is not None:
        if os.path.exists(repo_url.split("/")[-1]):
            os.system(f"rm -rf {repo_url.split('/')[-1]}")
        try:
            documents.extend(
                GitLoader(
                    clone_url=repo_url,
                    repo_path=f"./{repo_url.split('/')[-1]}",
                    branch=branch,
                ).load()
            )
        except:
            logging.log(logging.WARNING, f"Failed in cloning repo: {repo_url}")
            pass

    if file_dir is not None:
        try:
            documents.extend(DirectoryLoader(path=file_dir).load())
        except:
            pass

    if files is not None:
        for file in files:
            file_format = file.split(".")[-1]
            if file_format == "pdf":
                try:
                    documents.extend(PyPDFLoader(file_path=file).load())
                except:
                    logging.log(logging.WARNING, f"Failed in loading pdf: {file}")
                    pass
            elif file_format == "py":
                try:
                    documents.extend(PythonLoader(file_path=file).load())
                except:
                    logging.log(
                        logging.WARNING, f"Failed in loading python file: {file}"
                    )
                    pass
            elif file_format == "csv":
                try:
                    documents.extend(CSVLoader(file_path=file).load())
                except:
                    logging.log(logging.WARNING, f"Failed in loading csv file: {file}")
                    pass
            else:
                try:
                    documents.extend(TextLoader(file_path=file).load())
                except:
                    logging.log(logging.WARNING, f"Failed in loading file: {file}")
                    pass

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    docsearch = Chroma.from_documents(texts, embeddings)
    # make chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=get_llm(llm_type="openai", temperature=0.9, n=1, max_tokens=-1),
        retriever=docsearch.as_retriever(),
    )
    return qa
