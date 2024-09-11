apt update && apt install ffmpeg -y
git clone https://github.com/facebookresearch/co-tracker
cd co-tracker
pip install -e .
pip install opencv-python==4.7.0.72 einops timm matplotlib moviepy flow_vis 
pip install hydra-core==1.1.0 mediapy 
pip install streamlit_image_coordinates

mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth
cd ..

wget -P assets https://storage.googleapis.com/dm-tapnet/horsejump-high.mp4
