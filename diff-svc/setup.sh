#/bin/bash

pip install -r requirements.txt
apt update
DEBIAN_FRONTEND=noninteractive apt -y install ffmpeg