import argparse

from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, GitLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_url", type=str, default=None)
    parser.add_argument("--repo_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")



def make_qa_from_git(args) :

    loader = GitLoader(
        clone_url="https://github.com/vessl-ai/gitbook-docs.git",
        repo_path="./example/gitbook-docs",
        # branch="main",
    )

    ## git -> text
    documents = loader.load()

    ## text -> splitted text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    ## splitted text -> vector -> db
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)

    # make chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.9),
                                     chain_type="stuff",
                                     retriever=docsearch.as_retriever()
                                     )

    return qa



if __name__ == "__main__":


    args = parse_args()
    qa = make_qa_from_git(args)

    while (True) :
        if (input("exit?") == "y") :
            print("restart if you want to use again")
            break
        query = input("query : ")
        print(qa.run(query))

    print("done")


