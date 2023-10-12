import streamlit as st
import os, sys
import subprocess
from PIL import Image

st.set_page_config(layout="wide")

image_path = "./mistral-7B-v0.1.jpg"

@st.cache_data
def bind_socket():
    process=f"pip install -r requirements.txt"
    subprocess.run(process,shell=True)

col1, col2 = st.columns(2)

with col1:
    input_img = Image.open(image_path)
    st.image(input_img)
with col2:
    input_query = st.text_input('Input Query', 'Tom see flowers')
    bind_socket()
    st.write('Input: ' + input_query)
    input_query = input_query.encode()
    
    process=f"python main.py interactive /root/ckpt/mistral-7B-v0.1/ --max_tokens 256 --temperature 1.0 --prompt '{input_query}'"
    result = subprocess.run(process,shell=True, input=input_query, stdout=subprocess.PIPE)
    output_sentences = result.stdout.decode('utf-8')
    print(output_sentences)
    st.write(output_sentences)
