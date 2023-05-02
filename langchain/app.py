import streamlit as st
from test import *
@st.cache_resource()
def make_qa_from_git_streamlit() :
    return make_qa_from_git(None)

# Streamlit 앱 시작
qa = make_qa_from_git_streamlit()

st.title("Vessl docs searcher")
st.write("You can get answers by question and answering from vessl docs.")

# QA 모델 불러오기
st.write("Loading QA model...")
st.write("Done!")
# 사용자로부터 질문 입력 받기
user_question = st.text_input("Enter Question:")

# 질문 제출 버튼
submit_button = st.button("Submit Question")

if submit_button and user_question:
    # 질문에 대한 답변 얻기
    st.write("Question: " + user_question)

    st.write("Processing..")
    answer = qa.run(user_question)

    st.write(answer)
