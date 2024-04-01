import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

VESSL_LOGO_URL = (
    "https://vessl-public-apne2.s3.ap-northeast-2.amazonaws.com/vessl-logo/new_vessl-ai_color.png"
)

st.set_page_config(layout="wide")
st.image(VESSL_LOGO_URL, width=400)
intro = [
    "Manage your own Segment Anything session!",
    'One major hurdle in machine learning projects is establishing an environment. <a href="https://vessl.ai/floyd">VESSL AI</a> provides a solution to this bottleneck through YAML configuration. Using YAML configuration for a machine learning can offer a number of benefits:',
    "♻️ <strong>Reproducibility</strong>: Clearly define and save configurations as a file ensures that your experiments can be reproduced exactly.",
    "😉 <strong>Ease of Use</strong>: YAML files use a straightforward text format. This makes it easy for you to understand and modify the configurations as needed",
    "🚀 <strong>Scalability</strong>: A consistent method of using YAML files can be easily version-controlled, shared, and reused, which simplifies scaling.",
    "Try your own Segment Anything session with simple yaml we provide.",
]

st.title("Manage your own Segment Anything session!")
for e in intro:
    _e = f'<p style="font-family:system-ui; color:Black; font-size: 20px;">{e}</p>'
    st.markdown(_e, unsafe_allow_html=True)


@st.cache_data()
def mask_generate():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def show_anns(anns, ax):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    for ann in sorted_anns:
        m = ann["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


image_path = st.file_uploader("Upload your image!", type=["png", "jpg", "jpeg"])
if image_path is not None:
    with st.spinner("Running segmentation process ..."):
        image = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_generator = mask_generate()
        masks = mask_generator.generate(image)

        col1, col2, col3 = st.columns(3)

        with col2:
            o_fig, o_ax = plt.subplots(figsize=(10, 10))
            o_ax.imshow(image)
            o_ax.axis("off")
            st.pyplot(o_fig)
            st.success("Original Image")

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image)
            show_anns(masks, ax)
            ax.axis("off")
            st.pyplot(fig)
            st.success("Output Image")

col4, col5 = st.columns(2)
with col4:
    yaml = """name : segment-anything
description: "Segment ‘Anything’ using FAIR’s SAM with an interactive run on VESSL."
resources:
  cluster: aws-apne2
  preset: v1.v100-1.mem-52
image: nvcr.io/nvidia/pytorch:21.05-py3
run:
  - workdir: /root/examples/segment-anything/
    command: |
      bash ./setup.sh
import:
  /root/segment-anything: git://github.com/vessl-ai/examples
interactive:
  max_runtime: 24h
  jupyter:
    idle_timeout: 120m
ports:
  - 8501 """
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
    st.code(
        "pip install vessl\nvessl run create -f segment-anything.yaml",
        language="bash",
        line_numbers=False,
    )

st.markdown(
    f'<p style="font-family:system-ui; color:Black; font-size: 20px;">For further details, visit <a href="https://vesslai.mintlify.app/docs/reference/yaml">VESSL Run Docs</a>',
    unsafe_allow_html=True,
)
