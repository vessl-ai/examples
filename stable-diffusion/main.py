import streamlit as st
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from constants import INTRO, PREFIX, SUFFIX, VESSL_LOGO_URL

st.set_page_config(layout="wide")
st.image(VESSL_LOGO_URL, width=400)
intro = INTRO

st.title("Manage your own Stable Diffusion session!")
for e in intro:
    _e = f"{PREFIX}{e}{SUFFIX}"
    st.markdown(_e, unsafe_allow_html=True)

# Load model
model_id = "stabilityai/stable-diffusion-2-1"
# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

with st.form("prompt", clear_on_submit=False):
    prompt = st.text_area("Write down your prompt here: ", value="")
    submit_button = st.form_submit_button(label="Enter")
    if submit_button:
        image = pipe(prompt).images[0]

    if submit_button:
        st.image(image)

with open("stable-diffusion.yaml", "r") as f:
    yaml = f.read()
left, right = st.columns(2)
with left:
    st.markdown(
        f"{PREFIX}Here is the YAML we used for setting up this streamlit session.{SUFFIX}",
        unsafe_allow_html=True,
    )
    st.code(yaml, language="yaml", line_numbers=False)
with right:
    st.markdown(
        f"{PREFIX}You can save the YAML into a file and run it by yourself! Try:{SUFFIX}",
        unsafe_allow_html=True,
    )
    st.code(
        "pip install vessl\nvessl run -f stable-diffusion.yaml",
        language="bash",
        line_numbers=False,
    )

st.markdown(
    f'{PREFIX}For further details, visit <a href="https://vesslai.mintlify.app/docs/reference/yaml">VESSL Run Docs</a>.',
    unsafe_allow_html=True,
)
