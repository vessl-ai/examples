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
    'One major hurdle in machine learning projects is establishing an environment. <a href="https://vessl.ai/floyd">VESSL AI</a> provides a solution to this bottleneck through YAML configuration. Using YAML configuration for a machine learning can offer a number of benefits:',
    '♻️ <strong>Reproducibility</strong>: Clearly define and save configurations as a file ensures that your experiments can be reproduced exactly.',
    '😉 <strong>Ease of Use</strong>: YAML files use a straightforward text format. This makes it easy for you to understand and modify the configurations as needed',
    '🚀 <strong>Scalability</strong>: A consistent method of using YAML files can be easily version-controlled, shared, and reused, which simplifies scaling.',
    'Try your Stable Diffusion session using the simple yaml we offer.',
]

st.title("Manage your own Stable Diffusion session!")
for e in intro:
    _e = f'<p style="font-family:system-ui; color:Black; font-size: 20px;">{e}</p>'
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
        col1, col2, col3 = st.columns(3)

        with col2:
            st.image(image)

col4, col5 = st.columns(2)
with col4:
    yaml = """name : stable-diffusion
resources:
  cluster: aws-uw2-prod1
  accelerators: V100:1
image: quay.io/vessl-ai/ngc-pytorch-kernel:22.12-py3-202301160809
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
        f'<p style="font-family:system-ui; color:Black; font-size: 20px;">Here is the YAML we used for setting up this streamlit session.</p>',
        unsafe_allow_html=True,
    )
    st.code(yaml, language="yaml", line_numbers=False)
with col5:
    st.markdown(
        f'<p style="font-family:system-ui; color:Black; font-size: 20px;">You can save the YAML into a file and run it by yourself! Try:</p>',
        unsafe_allow_html=True,
    )
    st.code("pip install vessl\nvessl run -f stable-diffusion.yaml", language="bash", line_numbers=False)

st.markdown(
    f'<p style="font-family:system-ui; color:Black; font-size: 20px;">For further details, visit <a href="https://vesslai.mintlify.app/docs/reference/yaml">VESSL Run Docs</a>.',
    unsafe_allow_html=True,
)
