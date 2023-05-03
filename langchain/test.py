import argparse

from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, GitLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import DeepLake

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from repo2txt import get_docs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_url", type=str, default=None)
    parser.add_argument("--repo_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--mode", type=str, default="git")
    return parser.parse_args()


def filter(x):
    # filter based on source code
    if 'com.google' in x['text'].data()['value']:
        return False

    # filter based on path e.g. extension
    metadata = x['metadata'].data()['value']
    return 'scala' in metadata['source'] or 'py' in metadata['source']



def make_qa_from_git(args) :


    print("args: ", args)

    if args.repo_url is None and args.repo_dir is None :
        raise Exception("repo_url or repo_dir must be not None")

    if args.mode == "git" :
        loader = GitLoader(
            clone_url=args.repo_url,
            repo_path=args.repo_dir,
            # branch="main",
        )
        documents = loader.load()

        ## text -> splitted text
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(texts, embeddings)
        # make chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.9),
                                         chain_type="stuff",
                                         retriever=docsearch.as_retriever()
                                         )
        return qa

    elif args.mode == "text" :
        documents = get_docs(args.repo_dir, args.repo_url)

        ## text -> splitted text
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        username = "jinpil"  # replace with your username from app.activeloop.ai
        db = DeepLake(dataset_path=f"hub://{username}/{args.repo_dir.split('/')[-1]}", embedding_function=embeddings,
                      )
        db.add_documents(texts)

        retriever = db.as_retriever()
        retriever.search_kwargs['distance_metric'] = 'cos'
        retriever.search_kwargs['fetch_k'] = 100
        retriever.search_kwargs['maximal_marginal_relevance'] = True
        retriever.search_kwargs['k'] = 10
        retriever.search_kwargs['filter'] = filter

        model = ChatOpenAI(model='gpt-3.5-turbo')  # switch to 'gpt-4'
        qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
        return qa

if __name__ == "__main__":


    args = parse_args()
    qa = make_qa_from_git(args)
    history = []
    while (True) :
        if (input("exit? ") == "y") :
            print("restart if you want to use again")
            break

        elif (input("\nclear? ") == "y") :
            history = []
            print("history cleared")
            continue
        query = input("\nquery : ")
        result = qa({"question" : query, "chat_history" : history})
        print("question : ", result["question"])
        print("answer : ", result["answer"], "\n")
        history.append((query, result["answer"]))

    print("done")


