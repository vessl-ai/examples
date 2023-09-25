import streamlit as st
import subprocess
from PIL import Image

st.set_page_config(layout="wide")

image_path = "llama2_c/streamlit/llama_cute.jpg"
model_path = "/input/model.bin"


@st.cache_data
def prepare():
    subprocess.run(f"cd ./llama2_c && gcc -O3 -o run run.c -lm && chmod u+x {model_path}", shell=True)


col1, col2 = st.columns(2)

with col1:
    input_img = Image.open(image_path)
    st.image(input_img)

with col2:
    temperature = st.text_input('Temperature', '0.8')
    steps = st.text_input('Steps', '256')
    person_1 = st.text_input('The first hobbit\'s name', 'David')
    st.write('The name of the hobbit is', person_1)
    person_2 = st.text_input('The second hobbit\'s name', 'Floyd')
    st.write('A long time ago in a galaxy far, far away....')
    input_query = 'One day, ' + person_1 + ' met a ' + person_2

    prepare()

    process = f"cd ./llama2_c && ./run {model_path} -t {temperature} -n {steps} -i '{input_query}'"
    result = subprocess.run(process, shell=True, stdout=subprocess.PIPE)
    result.stdout.decode('utf-8')
    st.write(result.stdout.decode('utf-8'))
