import cv2
import warnings
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

VESSL_LOGO_URL = (
    "https://vessl-public-apne2.s3.ap-northeast-2.amazonaws.com/vessl-logo/vessl-ai_color_light"
    "-background.png"
)

st.set_page_config(layout="wide")
st.image(VESSL_LOGO_URL, width=400)
intro = [
    "VESSL AI: Manage your own Segment Anything session!",
    "Setting environment is one of the biggest bottleneck for machine learning projects. 😥",
    "VESSL AI lets you overcome the bottleneck with the use of yaml. 📋",
    "By providing yaml, you can declaratively run your machine learning projects in reliable manner!",
    "Try your own Segment Anything session with simple yaml we provide. 🚀",
]

st.title("VESSL AI: Manage your own Segment Anything session!")
for e in intro:
    _e = f'<p style="font-family:Courier; color:Black; font-size: 20px;">{e}</p>'
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
resources:
  cluster: aws-apne2-prod1
  accelerators: V100:1
image: nvcr.io/nvidia/pytorch:21.05-py3
run:
  - workdir: /root/segment-anything/
    command: |
      bash ./setup.sh
volumes:
  /root/segment-anything: git://github.com/saeyoon17/segment-anything
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
