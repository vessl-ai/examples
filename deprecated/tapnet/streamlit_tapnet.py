import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import os, sys
import subprocess
import mediapy as media
from PIL import Image, ImageDraw

def sample_random_points(height, width, num_points):
  """Sample random points with (time, height, width) order."""
  y = np.random.randint(0, height, (num_points, 1))
  x = np.random.randint(0, width, (num_points, 1))
  points = np.concatenate((y, x), axis=-1).astype(np.int32)
  return points

st.set_page_config(layout="wide")

video_path = 'tapnet/examplar_videos/horsejump-high.mp4'
video = media.read_video(video_path)

@st.cache_data
def bind_socket():
    process=f""
    subprocess.run(process,shell=True)

if "points" not in st.session_state:
    random_points = sample_random_points(video.shape[1], video.shape[2], 20)
    st.session_state["points"] = random_points
    
col1, col2 = st.columns(2)

with col1:
    frame0_path = 'tapnet/video_sample.png'
    media.write_image(frame0_path, video[0])
    with Image.open(frame0_path) as img:
        draw = ImageDraw.Draw(img)
        
        # Draw an ellipse at each coordinate in points
        for point in st.session_state["points"]:
            draw.ellipse([(point[0]-3,point[1]-3),(point[0]+3,point[1]+3)], fill="red")

        value = streamlit_image_coordinates(img, key="pil")

        if value is not None:
            point = value["x"], value["y"]
            if point not in st.session_state["points"]:
                st.session_state["points"] = np.append(st.session_state["points"], [point], axis=0)
                st.experimental_rerun()
with col2:
    if st.button('Add more points, and Click for doing inference!'):
        st.write("Inference working.. Please wait moment")
        np.save('selected_points.npy', st.session_state["points"])
        
        process=f"python3 infer_tapnet.py"
        result = subprocess.run(process,shell=True, stdout=subprocess.PIPE)
        result.stdout.decode('utf-8')
        st.write(result.stdout.decode('utf-8'))
        
        result_video_path = 'tapnet_sample_result.mp4'
        video_file = open(result_video_path, 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)
    else:
        st.write("Click button to inference")

    