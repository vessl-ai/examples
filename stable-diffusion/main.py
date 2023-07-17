import torch
import streamlit as st
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

# Load model from Hugging Face Diffusers
model_id = "stabilityai/stable-diffusion-2-1"
# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Configure page layout with Streamlit
st.set_page_config(layout="wide")

st.title("Stable Diffusion on VESSL Run")

with st.form("prompt", clear_on_submit=False):
    prompt = st.text_area("Enter your prompt here: ", value="")
    submit_button = st.form_submit_button(label="Enter")
    if submit_button:
        image = pipe(prompt).images[0]
    if submit_button:
        st.image(image)
