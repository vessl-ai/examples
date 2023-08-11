import diffusers
import torch
import os
import time
import accelerate
import streamlit as st

from stqdm import stqdm
from diffusers import DiffusionPipeline, UNet2DConditionModel
from PIL import Image


MODEL_REPO = "OFA-Sys/small-stable-diffusion-v0"
LoRa_DIR = "/ckpt"
DATASET_REPO = "VESSL/Bored_Ape_NFT_text"


def load_pipeline_w_lora():

    # Load pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        MODEL_REPO,
        revision=None,
        torch_dtype=torch.float32,
    )

    # Load LoRa attn layer weights to unet attn layers
    print("LoRa layers loading...")
    pipeline.unet.load_attn_procs(LoRa_DIR)
    print("LoRa layers loaded")

    pipeline.set_progress_bar_config(disable=True)

    return pipeline


# Configure page layout with Streamlit
st.set_page_config(layout="wide")

st.title("BAYC Text to IMAGE generator")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipeline = load_pipeline_w_lora().to(device)

col1, col2 = st.columns(2)

with col1:
    st.header("Prompt")
    with st.form("prompt", clear_on_submit=False):
        prompt = st.text_area("Write prompt to generate your unique BAYC image! (e.g. An ape with golden fur)")
        num_images = st.number_input(label="Number of images to generate", min_value=1, max_value=10)

        seed = st.number_input(label="Seed for images", min_value=1, max_value=10000)
        submit_button = st.form_submit_button(label="Generate")

with col2:
    st.header("Image")
    st.write(f"Generating {num_images} BAYC image with prompt <{prompt}>...")

    if submit_button:
        generator = torch.Generator(device=device).manual_seed(seed)
        images = []
        for img_idx in stqdm(range(num_images)):
            generated_image = pipeline(prompt, num_inference_steps=30, generator=generator).images[0]
            images.append(generated_image)

        st.write("Done!")

        st.image(images, width=300, caption=[f"Generated Images with <{prompt}>" for i in range(len(images))])

st.markdown("Explore more models at [vessl.ai/hub](https://vessl.ai/).")
st.image("https://i.imgur.com/UpdYC1d.png", width=180)
