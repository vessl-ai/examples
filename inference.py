import os
import torch

from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
import mediapy as media
import numpy as np

video_path = 'co-tracker/assets/horsejump-high.mp4'
video = media.read_video(video_path)
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

from cotracker.predictor import CoTrackerPredictor

model = CoTrackerPredictor(
    checkpoint=os.path.join(
        './co-tracker/checkpoints/cotracker_stride_4_wind_8.pth'
    )
)

select_points = np.load('selected_points.npy')
queries = torch.tensor(select_points)
# queries = torch.tensor([
#     [0., 100., 100.],  # point tracked from the first frame
#     [5., 150., 150.], # frame number 5
#     [10., 200., 200.]
# ])

if torch.cuda.is_available():
    model = model.cuda()
    video = video.cuda()
    queries = queries.cuda()

# pred_tracks, pred_visibility = model(video, grid_size=30)
pred_tracks, pred_visibility = model(video, queries=queries[None])

vis = Visualizer(
    save_dir='./pred_videos',
    linewidth=3,
    mode='cool',
    tracks_leave_trace=-1
)
vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename='queries');

