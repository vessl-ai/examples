import streamlit as st
import os, sys
import subprocess
from PIL import Image
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained("/data/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# To run streamlit, setup streamlit port as 80 and run 'streamlit run SSD-1B_streamlit.py --server.port=80'

st.set_page_config(layout="wide")

image_path = "./An astronaut riding a green horse.jpg"

col1, col2 = st.columns(2)

with col1:
    input_img = Image.open(image_path)
    st.image(input_img)
with col2:
    prompt = st.text_input('Input Query', 'Tom see flowers')
    bind_socket()
        
    prompt = "An astronaut riding a green horse" # Your prompt here
    neg_prompt = "ugly, blurry, poor quality, scary" # Negative prompt here
    image = pipe(prompt=prompt, negative_prompt=neg_prompt).images[0]
    image.save(prompt + '.jpg')
    
    print(output_sentences)
    st.write(output_sentences)
