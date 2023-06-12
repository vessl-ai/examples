import torch
import streamlit as st
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

VESSL_LOGO_URL = (
    "https://vessl-public-apne2.s3.ap-northeast-2.amazonaws.com/vessl-logo/vessl-ai_color_light"
    "-background.png"
)

st.set_page_config(layout="wide")
st.image(VESSL_LOGO_URL, width=400)
intro = [
    "VESSL AI: Manage your own Stable Diffusion session!",
    "Setting environment is one of the biggest bottleneck for machine learning projects. 😥",
    "VESSL AI lets you overcome the bottleneck with the use of yaml. 📋",
    "By providing yaml, you can declaratively run your machine learning projects in reliable manner!",
    "Try your own Stable Diffusion session with simple yaml we provide. 🚀",
]

st.title("VESSL AI: Manage your own Stable Diffusion session!")
for e in intro:
    _e = f'<p style="font-family:Courier; color:Black; font-size: 20px;">{e}</p>'
    st.markdown(_e, unsafe_allow_html=True)


# Load model
model_id = "stabilityai/stable-diffusion-2-1"
# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

with st.form("prompt", clear_on_submit=False):
    prompt = st.text_area("Write down your prompt here!: ", value="")
    submit_button = st.form_submit_button(label="Enter")
    if submit_button:
        image = pipe(prompt).images[0]

    if submit_button:
        col1, col2, col3 = st.columns(3)

        with col2:
            st.image(image)

col4, col5 = st.columns(2)
with col4:
    yaml = """name : stable-diffusion
resources:
  cluster: aws-apne2-prod1
  accelerators: V100:1
image: nvcr.io/nvidia/pytorch:21.05-py3
run:
  - workdir: /root/stable-diff-st/
    command: |
      bash ./setup.sh
volumes:
  /root/stable-diff-st: git://github.com/saeyoon17/stable-diff-st
interactive:
  runtime: 24h
  ports:
    - 8501
    """
    st.markdown(
        f'<p style="font-family:Courier; color:Black; font-size: 20px;">Here, we provide the yaml we used for setting up this streamlit session.</p>',
        unsafe_allow_html=True,
    )
    st.code(yaml, language="yaml", line_numbers=False)
with col5:
    st.markdown(
        f'<p style="font-family:Courier; color:Black; font-size: 20px;">You can save the yaml and run it by userself! Try:</p>',
        unsafe_allow_html=True,
    )
    st.code("pip install vessl\nvessl run -f youryaml", language="python", line_numbers=False)

st.markdown(
    f'<p style="font-family:Courier; color:Black; font-size: 20px;">For further details, visit <a href="https://vessl.ai/">link!</a>',
    unsafe_allow_html=True,
)
