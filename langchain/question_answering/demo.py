import streamlit as st
from utils import *
from streamlit_chat import message as msg
import os

VESSL_LOGO_URL = "https://vessl-public-apne2.s3.ap-northeast-2.amazonaws.com/vessl-logo/vessl-ai_color_light" \
                 "-background.png"
st.set_page_config(layout="wide")


@st.cache_resource()
def MakingQA(repo_url=None, branch="main", files=None, chunk_size=500, chunk_overlap=100):
    st.write("Making QA Chatbot Server from your data with VESSL, it may take a few minutes depends on your data size")
    return make_qa(repo_url,
                   branch,
                   files,
                   file_dir=None,
                   chunk_size=chunk_size,
                   chunk_overlap=chunk_overlap,
                   )


def streamlit_display_title():
    st.image(VESSL_LOGO_URL, width=400)
    st.title("Making QA Chatbot Server from your data with VESSL")
    st.write("VESSL is End-to-End MLops platform for ML engineers. Please check our product at https://vessl.ai/")
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
            st.write("click submit button again to move next step")
            quit()

    if "api_key" not in st.session_state:
        st.warning("Please enter your OpenAI API key")
        quit()

    if "uploaded_files" not in st.session_state and "git_repo" not in st.session_state:
        st.subheader("2. Please upload your files or enter git repository url")
        uploaded_files = st.file_uploader("Upload files",
                                          accept_multiple_files=True,
                                          type=["txt", "pdf", "docx", "py", "c", "h"]
                                          )
        repo_url = st.text_input("Enter git repo url to integrate in your source", value="https://github.com/vessl-ai"
                                                                                         "/gitbook-docs.git")
        repo_branch = st.text_input("Enter branch of repo url to integrate in your source", value="main")
        chunk_size = st.number_input("Enter chunk size [How large to split long texts]", value=500)
        chunk_overlap = st.number_input("Enter chunk overlap [How much to overlap splitted texts]", value=100)
        submit_button = st.button("Submit", key="submit_button")
        if submit_button and (uploaded_files or repo_url) and chunk_size and chunk_overlap:
            file_paths = []
            if uploaded_files:
                os.makedirs("./streamlit_data", exist_ok=True)
                for uploaded_file in uploaded_files:
                    with open(os.path.join("./streamlit_data", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(os.path.join("./streamlit_data", uploaded_file.name))
                    st.success("Saved file: {}".format(uploaded_file.name))

            st.session_state["uploaded_files"] = file_paths
            st.session_state["git_repo"] = repo_url
            st.session_state["git_branch"] = repo_branch
            st.session_state["chunk_size"] = chunk_size
            st.session_state["chunk_overlap"] = chunk_overlap
            st.write("click submit button again to move next step")
            quit()

    if st.session_state.get('uploaded_files') is None and st.session_state.get('git_repo') is None:
        st.warning("Please upload files or enter git repo url")
        quit()

    qa = MakingQA(
        files=st.session_state["uploaded_files"],
        repo_url=st.session_state["git_repo"],
        branch=st.session_state["git_branch"],
        chunk_size=st.session_state["chunk_size"],
        chunk_overlap=st.session_state["chunk_overlap"],
    )

    user_question = st.text_input("Enter Question ('clear' to reset history)")
    submit_button = st.button("Submit Question")

    if "history" not in st.session_state:
        st.session_state['history'] = []

    # ensure displaying history
    for hist in st.session_state['history']:
        msg(hist[0], is_user=True)
        msg(hist[1], is_user=False)

    if submit_button and user_question:
        if user_question == "clear":
            st.session_state['history'] = []
            st.success("History cleared")
            quit()
        msg(user_question, is_user=True)
        try:
            msg("Processing your question...", is_user=False)
            result = qa({"question": user_question, "chat_history": st.session_state['history']})
        except:
            msg("Error occured while processing, please retry with fresh question", is_user=False)
            return
        st.session_state['history'].append((user_question, result["answer"]))
        # display new answer
        msg(result["answer"], is_user=False)


if __name__ == "__main__":
    main()
