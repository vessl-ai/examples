import getpass
import os

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

# os.environ["OPENAI_API_KEY"] = getpass.getpass()
model = ChatOpenAI(base_url="http://gemma2-ko.vessl.ai/v1", model="/root/model/merged", temperature=0, max_tokens=500)



# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory

# store = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# with_message_history = RunnableWithMessageHistory(model, get_session_history)

# config = {"configurable": {"session_id": "abc2"}}

# response = with_message_history.invoke(
#     [HumanMessage(content="Hi! I'm Bob")],
#     config=config,
# )

# response.content

def respond(message, history): 
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = model(history_langchain_format)
    return gpt_response.content

gr.ChatInterface(fn=respond, title="Chat with gemma2-kor").launch()