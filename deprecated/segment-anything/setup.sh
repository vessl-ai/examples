pip install accelerate
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install streamlit
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
streamlit run infer_streamlit.py