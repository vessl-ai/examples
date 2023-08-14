import os
from PIL import Image
import streamlit as st

MODEL_REPO = "OFA-Sys/small-stable-diffusion-v0"
LoRa_DIR = "/ckpt"
DATASET_REPO = "VESSL/Bored_Ape_NFT_text"

# Configure page layout with Streamlit
st.set_page_config(layout="wide")
st.title("Image animation using Thin-Plate-Spline-Motion-Model")


def inference(vid):
    os.system(f"python demo.py --config config/vox-256.yaml --checkpoint /ckpt/vox.pth.tar --source_image 'temp/image.jpg' --driving_video {vid} --result_video './temp/result.mp4'")
    return "./temp/result.mp4"


col1, col2 = st.columns(2)

with col1:
    st.header("Upload image or Take your photo!")
    with st.form("image", clear_on_submit=False):
        image_path = st.file_uploader("Upload your image!", type=["png", "jpg", "jpeg"])

        img_file_buffer = st.camera_input("Take your photo!")
        if not os.path.exists("temp"):
            os.system("mkdir temp")
        if img_file_buffer != None:
            img = Image.open(img_file_buffer)
        else:
            img = Image.open(image_path)
        img.save("temp/image.jpg", "JPEG")
        submit_button = st.form_submit_button(label="Generate")

    st.header("Driving Video")
    video_file = open("./assets/driving.mp4", "rb")
    video_bytes = video_file.read()
    st.video([video_bytes, video_bytes, video_bytes])
    option = st.radio("Choose your driving video!", ["Video 1", "Video 2", "Video 3"])

with col2:
    st.header("Result Video")
    if submit_button:
        result_video = inference("./assets/driving.mp4")
        video_file = open(result_video, "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)

st.markdown("Explore more models at [vessl.ai/hub](https://vessl.ai/).")
st.image("https://i.imgur.com/UpdYC1d.png", width=180)
