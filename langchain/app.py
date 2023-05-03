import streamlit as st
from test import *
@st.cache_resource()
def make_qa_from_git_streamlit() :
    return make_qa_from_git(None)

def main() :
    qa = make_qa_from_git_streamlit()

    st.title("Vessl docs searcher")
    st.write("You can get answers by question and answering from vessl docs.")

    user_question = st.text_input("Enter Question:")
    submit_button = st.button("Submit Question")

    if submit_button and user_question:
        # 질문에 대한 답변 얻기
        st.write("Question: " + user_question)
        st.write("Processing..")
        answer = qa.run(user_question)
        st.write(answer)

if __name__ == "__main__" :
    main()
