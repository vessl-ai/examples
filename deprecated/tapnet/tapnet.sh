apt update && apt install ffmpeg -y
pip3 install streamlit_image_coordinates

git clone https://github.com/deepmind/tapnet.git
pip install -r tapnet/requirements_inference.txt

mkdir tapnet/checkpoints
wget -P tapnet/checkpoints https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy

mkdir tapnet/examplar_videos
wget -P tapnet/examplar_videos https://storage.googleapis.com/dm-tapnet/horsejump-high.mp4

pip install -U jax==0.4.11 jaxlib==0.4.11+cuda12.cudnn88 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
