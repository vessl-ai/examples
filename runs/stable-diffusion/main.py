import streamlit as st
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

# Load model from Hugging Face Diffusers
model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Configure page layout with Streamlit
st.set_page_config(layout="wide")

st.title("Stable Diffusion with VESSL Run")

col1, col2 = st.columns(2)

with col1:
    st.header("Prompt")
    with st.form("prompt", clear_on_submit=False):
        prompt = st.text_area("Enter your prompt here")
        submit_button = st.form_submit_button(label="Generate")

with col2:
    st.header("Image")
    if submit_button:
        image = pipe(prompt).images[0]
    if submit_button:
        st.image(image)

st.markdown("Explore more models at [vessl.ai/hub](https://app.vessl.ai/).")
st.image("https://i.imgur.com/UpdYC1d.png", width=180)
