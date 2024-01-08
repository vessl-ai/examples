import os

import streamlit as st
from PIL import Image

MODEL_REPO = "OFA-Sys/small-stable-diffusion-v0"
LoRa_DIR = "/ckpt"
DATASET_REPO = "VESSL/Bored_Ape_NFT_text"

# Configure page layout with Streamlit
st.set_page_config(layout="wide")
st.title("Image animation using Thin-Plate-Spline-Motion-Model")


def inference(vid):
    os.system(
        f"python demo.py --config config/vox-256.yaml --checkpoint /ckpt/vox.pth.tar --source_image 'temp/image.jpg' --driving_video {vid} --result_video './temp/result.mp4'"
    )
    return "./temp/result.mp4"


st.header("Use predefined driving video or upload your own!")
_col1, _col2 = st.columns(2)
video_file = open("./assets/driving.mp4", "rb")
video_bytes = video_file.read()
with _col1:
    st.video(video_bytes)
with _col2:
    with st.form("Driving video", clear_on_submit=False):
        video_path = st.file_uploader("Upload your own driving video!", type=["mp4"])
        video_submit_button = st.form_submit_button(label="Submit Driving Video")
if video_path == None:
    video_path = "./assets/driving.mp4"


col1, col2 = st.columns(2)
with col1:
    st.header("Upload image or Take your photo!")
    with st.form("image", clear_on_submit=False):
        image_path = st.file_uploader("Upload your image!", type=["png", "jpg", "jpeg"])
        img_file_buffer = st.camera_input("Take your photo!")
        submit_button = st.form_submit_button(label="Generate")
        if not os.path.exists("temp"):
            os.system("mkdir temp")
        if img_file_buffer != None:
            img = Image.open(img_file_buffer)
            img.save("temp/image.jpg", "JPEG")
        elif image_path != None:
            img = Image.open(image_path)
            img.save("temp/image.jpg", "JPEG")
with col2:
    st.header("Result Video")
    if submit_button:
        result_video = inference(video_path)
        video_file = open(result_video, "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)

st.markdown("Explore more models at [vessl.ai/hub](https://vessl.ai/).")
st.image("https://i.imgur.com/UpdYC1d.png", width=180)
