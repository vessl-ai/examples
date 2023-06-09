import cv2
import warnings
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

warnings.filterwarnings("ignore")
VESSL_LOGO_URL = (
    "https://vessl-public-apne2.s3.ap-northeast-2.amazonaws.com/vessl-logo/vessl-ai_color_light"
    "-background.png"
)


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


st.set_page_config(page_title="VESSL-AI: Segment Anything!", layout="wide")
st.image(VESSL_LOGO_URL, width=400)
st.write(
    "VESSL is End-to-End MLops platform for ML engineers. Please check our product at https://vessl.ai/"
)
st.info(
    "This project is ran using 'vessl run'. Check out other projects you can run through the link!"
)

image_path = st.file_uploader("Upload your image!", type=["png", "jpg", "jpeg"])
if image_path is not None:
    with st.spinner("Running segmentation process ..."):
        image = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_generator = mask_generate()
        masks = mask_generator.generate(image)

        col1, col2, col3 = st.columns(3)

        with col2:
            st.image(image)
            st.success("Original Image")

            fig, ax = plt.subplots(figsize=(15, 15))
            ax.imshow(image)
            show_anns(masks, ax)
            ax.axis("off")
            st.pyplot(fig)
            st.success("Output Image")
