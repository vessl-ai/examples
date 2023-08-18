import streamlit as st
import torch
from diffusers import DiffusionPipeline
from stqdm import stqdm

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
    pipeline.unet.load_attn_procs(LoRa_DIR)
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
        prompt = st.text_area(
            "Write prompt to generate your unique BAYC image! (e.g. An ape with golden fur)"
        )
        num_images = st.number_input(
            label="Number of images to generate", min_value=1, max_value=10
        )
        seed = st.number_input(label="Seed for images", min_value=1, max_value=10000)
        submit_button = st.form_submit_button(label="Generate")

with col2:
    st.header("Image")
    if submit_button:
        generator = torch.Generator(device=device).manual_seed(seed)
        images = []
        cols = st.columns(num_images)
        for img_idx in stqdm(range(num_images)):
            generated_image = pipeline(
                prompt, num_inference_steps=30, generator=generator
            ).images[0]
            images.append(generated_image)
            with cols[img_idx]:
                st.image(generated_image, width=200, caption=[f"Image: {img_idx}"])

st.markdown("Explore more models at [vessl.ai/hub](https://vessl.ai/).")
st.image("https://i.imgur.com/UpdYC1d.png", width=180)
