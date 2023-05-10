import streamlit as st
from utils import *
import streamlit_chat
import os
import sys
import time

VESSL_LOGO_URL = "https://vessl-public-apne2.s3.ap-northeast-2.amazonaws.com/vessl-logo/vessl_logo.jpeg"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_url", type=str, default=None)
    parser.add_argument("--repo_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--mode", type=str, default="git")
    parser.add_argument("--add", type=bool, default=False)
    return parser.parse_args()


@st.cache_resource()
def MakingQA(repo_url=None, branch="main", files=None, chunk_size=500, chunk_overlap=100):
    return make_qa(repo_url,
                   branch,
                   files,
                   file_dir=None,
                   chunk_size=chunk_size,
                   chunk_overlap=chunk_overlap,
                   )


@st.cache_data()
def QuestionProcessing(qa, question, chat_history):
    return qa({"question": question, "chat_history": chat_history})


def streamlit_display_title():
    st.image(VESSL_LOGO_URL, width=400)
    st.title("Langchain with VESSL")
    st.subheader("VESSL is End-to-End MLops platform for ML engineers. Please check our product at https://vessl.ai/")
    return


def main():
    streamlit_display_title()

    if "api_key" not in st.session_state:
        st.subheader("1. Please enter your OpenAI API key")
        api_key = st.text_input("OpenAI API key")
        submit_button = st.button("Submit")
        if submit_button and api_key:
            st.session_state["api_key"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key

    if "api_key" not in st.session_state:
        st.warning("Please enter your OpenAI API key")
        quit()

    if "uploaded_files" not in st.session_state and "git_repo" not in st.session_state:
        st.subheader("2. Please upload your files or enter git repository url")
        uploaded_files = st.file_uploader("Upload files",
                                          accept_multiple_files=True,
                                          type=["txt", "pdf", "docx", "py", "c", "h"]
                                          )
        repo_url = st.text_input("Enter git repo url to integrate in your source", value="")
        repo_branch = st.text_input("Enter branch of repo url to integrate in your source", value="main")
        chunk_size = st.number_input("Enter chunk size", value=500)
        chunk_overlap = st.number_input("Enter chunk overlap", value=100)
        submit_button = st.button("Submit", key="submit_button")
        if submit_button and (uploaded_files or repo_url) and chunk_size and chunk_overlap:
            file_paths = []
            if uploaded_files:
                os.makedirs("./tmp", exist_ok=True)
                for uploaded_file in uploaded_files:
                    with open(os.path.join("./tmp", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(os.path.join("./tmp", uploaded_file.name))
                    st.success("Saved file: {}".format(uploaded_file.name))

            st.session_state["uploaded_files"] = file_paths
            st.session_state["git_repo"] = repo_url
            st.session_state["git_branch"] = repo_branch
            st.session_state["chunk_size"] = chunk_size
            st.session_state["chunk_overlap"] = chunk_overlap
            # quit()

    if st.session_state.get('uploaded_files') is None and st.session_state.get('git_repo') is "":
        st.warning("Please upload files or enter git repo url")
        quit()

    qa = MakingQA(
        files=st.session_state["uploaded_files"],
        repo_url=st.session_state["git_repo"],
        branch=st.session_state["git_branch"],
        chunk_size=st.session_state["chunk_size"],
        chunk_overlap=st.session_state["chunk_overlap"],
    )

    user_question = st.text_input("Enter Question:")
    submit_button = st.button("Submit Question")

    if "history" not in st.session_state:
        st.session_state['history'] = []

    ## ensure displaying history
    for hist in st.session_state['history']:
        st.write("Q: " + hist[0])
        st.write("A: " + hist[1])
        st.write("")

    if submit_button and user_question:
        st.write("Q: " + user_question)
        try:
            result = QuestionProcessing(qa, user_question, st.session_state['history'])
        except:
            st.write("Error occured while processing, please retry with fresh question")
            # TODO: 에러 메세지 띄우기 + 핸들링
            return
        st.session_state['history'].append((user_question, result["answer"]))
        ## display new answer
        st.write("A: " + result["answer"])
        st.write("")


if __name__ == "__main__":
    main()
