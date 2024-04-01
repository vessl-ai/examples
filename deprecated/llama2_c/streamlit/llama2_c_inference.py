import streamlit as st
import subprocess
from PIL import Image

st.set_page_config(layout="wide")

image_path = "llama2_c/streamlit/llama_cute.jpg"
model_path = "/root/examples/llama2_c/stories42M.bin"


@st.cache_data
def prepare():
    subprocess.run(f"cd ./llama2_c && gcc -O3 -o run run.c -lm && chmod u+x {model_path}", shell=True)


col1, col2 = st.columns(2)

with col1:
    input_img = Image.open(image_path)
    st.image(input_img)

with col2:
    temperature = st.text_input('Temperature', '0.9')
    steps = st.text_input('Steps', '256')
    prompts = st.text_input('Prompts', 'One day, Lily met a Shoggoth')

    prepare()

    command = f"cd ./llama2_c && ./run {model_path} -t {temperature} -n {steps} -i \"{prompts}\""
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    result.stdout.decode('utf-8')
    st.write(result.stdout.decode('utf-8'))
