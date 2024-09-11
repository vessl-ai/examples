__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from streamlit_chat import message as msg
from utils import *

VESSL_LOGO_URL = (
    "https://vessl-public-apne2.s3.ap-northeast-2.amazonaws.com/vessl-logo/new_vessl-ai_color.png"
)
st.set_page_config(layout="wide")


@st.cache_resource()
def MakingQA(
    repo_url=None, branch="main", files=None, chunk_size=500, chunk_overlap=100
):
    st.write(
        "Making QA Chatbot Server from your data with VESSL, it may take a few minutes depends on your data size"
    )
    return make_qa(
        repo_url,
        branch,
        files,
        file_dir=None,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def streamlit_display_title():
    st.image(VESSL_LOGO_URL, width=400)
    intro = [
        'One major hurdle in machine learning projects is establishing an environment. <a href="https://vessl.ai/floyd">VESSL AI</a> provides a solution to this bottleneck through YAML configuration. Using YAML configuration for a machine learning can offer a number of benefits:',
        "♻️ <strong>Reproducibility</strong>: Clearly define and save configurations as a file ensures that your experiments can be reproduced exactly.",
        "😉 <strong>Ease of Use</strong>: YAML files use a straightforward text format. This makes it easy for you to understand and modify the configurations as needed",
        "🚀 <strong>Scalability</strong>: A consistent method of using YAML files can be easily version-controlled, shared, and reused, which simplifies scaling.",
        "Try your own LangChain session with simple yaml we provide.",
    ]

    st.title("Manage your own Langchain session!")
    for e in intro:
        _e = f'<p style="font-family:system-ui; color:Black; font-size: 20px;">{e}</p>'
        st.markdown(_e, unsafe_allow_html=True)
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
        uploaded_files = st.file_uploader(
            "Upload files",
            accept_multiple_files=True,
            type=["txt", "pdf", "docx", "py", "c", "h"],
        )
        repo_url = st.text_input(
            "Enter git repo url to integrate in your source",
            value="https://github.com/vessl-ai" "/gitbook-docs.git",
        )
        repo_branch = st.text_input(
            "Enter branch of repo url to integrate in your source", value="main"
        )
        chunk_size = st.number_input(
            "Enter chunk size [How large to split long texts]", value=500
        )
        chunk_overlap = st.number_input(
            "Enter chunk overlap [How much to overlap split texts]", value=100
        )
        submit_button = st.button("Submit", key="submit_button")
        if (
            submit_button
            and (uploaded_files or repo_url)
            and chunk_size
            and chunk_overlap
        ):
            file_paths = []
            if uploaded_files:
                os.makedirs("./streamlit_data", exist_ok=True)
                for uploaded_file in uploaded_files:
                    with open(
                        os.path.join("./streamlit_data", uploaded_file.name), "wb"
                    ) as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(
                        os.path.join("./streamlit_data", uploaded_file.name)
                    )
                    st.success("Saved file: {}".format(uploaded_file.name))

            st.session_state["uploaded_files"] = file_paths
            st.session_state["git_repo"] = repo_url
            st.session_state["git_branch"] = repo_branch
            st.session_state["chunk_size"] = chunk_size
            st.session_state["chunk_overlap"] = chunk_overlap
            st.write("click submit button again to move next step")
            quit()

    if (
        st.session_state.get("uploaded_files") is None
        and st.session_state.get("git_repo") is None
    ):
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
        st.session_state["history"] = []

    # ensure displaying history
    for hist in st.session_state["history"]:
        msg(hist[0], is_user=True)
        msg(hist[1], is_user=False)

    if submit_button and user_question:
        if user_question == "clear":
            st.session_state["history"] = []
            st.success("History cleared")
            quit()
        msg(user_question, is_user=True)
        try:
            msg("Processing your question...", is_user=False)
            result = qa(
                {"question": user_question, "chat_history": st.session_state["history"]}
            )
        except:
            msg(
                "Error occurred while processing, please retry with fresh question",
                is_user=False,
            )
            return
        st.session_state["history"].append((user_question, result["answer"]))
        # display new answer
        msg(result["answer"], is_user=False)

    col4, col5 = st.columns(2)
    with col4:
        yaml = """name : langchain
description: "Generate your own assistant trained on your data with an interactive run on VESSL."
resources:
  cluster: aws-apne2
  preset: v1.cpu-4.mem-13
image: quay.io/vessl-ai/kernels:py38-202303150331
run:
  - workdir: /root/examples/langchain/question_answering/
    command: |
      bash ./run.sh
import:
  /root/examples: git://github.com/vessl-ai/examples
interactive:
  max_runtime: 24h
  jupyter:
    idle_timeout: 120m
ports:
  - 8501
        """
        st.markdown(
            f'<p style="font-family:system-ui; color:Black; font-size: 20px;">Here is the YAML we used for setting up this streamlit session.</p>',
            unsafe_allow_html=True,
        )
        st.code(yaml, language="yaml", line_numbers=False)
    with col5:
        st.markdown(
            f'<p style="font-family:system-ui; color:Black; font-size: 20px;">You can save the YAML into a file and run it by yourself! Try:</p>',
            unsafe_allow_html=True,
        )
        st.code(
            "pip install vessl\nvessl run create -f langchain.yaml",
            language="bash",
            line_numbers=False,
        )

    st.markdown(
        f'<p style="font-family:system-ui; color:Black; font-size: 20px;">For further details, visit <a href="https://vesslai.mintlify.app/docs/reference/yaml">VESSL Run Docs</a>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
