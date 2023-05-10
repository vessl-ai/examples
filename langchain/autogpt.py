# script for langchain autogpt test


from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings

import faiss

from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI


if __name__ == "__main__" :

    search = SerpAPIWrapper()
    tools = [
        Tool(
            name = "search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]

    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()

    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    agent = AutoGPT.from_llm_and_tools(
        ai_name="Tommy",
        ai_role="Boss",
        tools=tools,
        llm=ChatOpenAI(temperature=0),
        memory=vectorstore.as_retriever(),
        human_in_the_loop=True,
    )

    # Set verbose to be true
    agent.chain.verbose = False

    # Run the agent with a query
    agent.run("What is the weather like today in Korea?")