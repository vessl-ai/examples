import streamlit as st
import torch
from DeepCache import DeepCacheSDHelper
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(
    cache_interval=3,
    cache_branch_id=0,
)

st.set_page_config(layout="wide")

st.title("Stable Diffusion with VESSL Run")

col1, col2 = st.columns(2)

with col1:
    st.header("Prompt")
    with st.form("prompt", clear_on_submit=False):
        prompt = st.text_area("Enter your prompt here")
        deepcache = st.toggle("Enable DeepCache", value=True)
        submit_button = st.form_submit_button(label="Generate")

with col2:
    st.header("Image")
    if submit_button:
        if deepcache:
            helper.enable()
        image = pipe(prompt).images[0]
        if deepcache:
            helper.disable()
    if submit_button:
        st.image(image)

st.markdown("Explore more models at [vessl.ai/hub](https://app.vessl.ai/hub).")
st.image("https://i.imgur.com/xkJwAxL.png", width=180)
