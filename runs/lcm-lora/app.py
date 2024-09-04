import streamlit as st
import torch
from diffusers import DiffusionPipeline, LCMScheduler

st.set_page_config(layout="wide")

st.title("LCM-LoRA on VESSL Run")


@st.cache_resource(show_spinner="Loading pipeline...")
def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        variant="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm-lora")

    return pipe, pipe.scheduler


pipe, scheduler = load_pipeline()
pipe.disable_lora()

col1, col2 = st.columns(2)

with col1:
    st.header("Prompt")
    with st.form("prompt", clear_on_submit=False):
        prompt = st.text_area(
            "Enter your prompt here",
            "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        )
        steps = st.number_input("Steps", 1, 8, 4, 1)
        guidance_scale = st.slider("Guidance Scale", 0.0, 2.0, 1.0, 0.1)
        enable_lcm = st.toggle("Enable LCM-LoRA", value=True)
        submit_button = st.form_submit_button(label="Generate")

with col2:
    st.header("Image")
    if submit_button:
        if enable_lcm:
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.enable_lora()
        image = pipe(
            prompt=prompt, num_inference_steps=steps, guidance_scale=guidance_scale
        ).images[0]
        if enable_lcm:
            pipe.scheduler = scheduler
            pipe.disable_lora()

    if submit_button:
        st.image(image)

st.markdown("Explore more models at [vessl.ai/hub](https://app.vessl.ai/hub).")
st.image("https://i.imgur.com/xkJwAxL.png", width=180)
