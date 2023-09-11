import haiku as hk
import jax
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import tree

from tapnet import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils

checkpoint_path = 'tapnet/checkpoints/tapir_checkpoint_panning.npy'
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state['params'], ckpt_state['state']

def build_model(frames, query_points):
  """Compute point tracks and occlusions given frames and query points."""
  model = tapir_model.TAPIR(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
  outputs = model(
      video=frames,
      is_training=False,
      query_points=query_points,
      query_chunk_size=64,
  )
  return outputs

model = hk.transform_with_state(build_model)
model_apply = jax.jit(model.apply)

def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.astype(np.float32)
  frames = frames / 255 * 2 - 1
  return frames


def postprocess_occlusions(occlusions, expected_dist):
  """Postprocess occlusions to boolean visible flag.

  Args:
    occlusions: [num_points, num_frames], [-inf, inf], np.float32
    expected_dist: [num_points, num_frames], [-inf, inf], np.float32

  Returns:
    visibles: [num_points, num_frames], bool
  """
  visibles = (1 - jax.nn.sigmoid(occlusions)) * (1 - jax.nn.sigmoid(expected_dist)) > 0.5
  return visibles

def inference(frames, query_points):
  """Inference on one video.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8
    query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

  Returns:
    tracks: [num_points, 3], [-1, 1], [t, y, x]
    visibles: [num_points, num_frames], bool
  """
  # Preprocess video to match model inputs format
  frames = preprocess_frames(frames)
  num_frames, height, width = frames.shape[0:3]
  query_points = query_points.astype(np.float32)
  frames, query_points = frames[None], query_points[None]  # Add batch dimension

  # Model inference
  rng = jax.random.PRNGKey(42)
  outputs, _ = model_apply(params, state, rng, frames, query_points)
  outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)
  tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']

  # Binarize occlusions
  visibles = postprocess_occlusions(occlusions, expected_dist)
  return tracks, visibles


def sample_random_points(frame_max_idx, height, width, num_points):
  """Sample random points with (time, height, width) order."""
  y = np.random.randint(0, height, (num_points, 1))
  x = np.random.randint(0, width, (num_points, 1))
  t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
  points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
  return points

def convert_select_points_to_query_points(frame, points):
  """Convert select points to query points.

  Args:
    points: [num_points, 2], in [x, y]
  Returns:
    query_points: [num_points, 3], in [t, y, x]
  """
  points = np.stack(points)
  query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
  query_points[:, 0] = frame
  query_points[:, 1] = points[:, 1]
  query_points[:, 2] = points[:, 0]
  return query_points

video = media.read_video('tapnet/examplar_videos/horsejump-high.mp4')
height, width = video.shape[1:3]

resize_height = 256  # @param {type: "integer"}
resize_width = 256  # @param {type: "integer"}

frames = media.resize_video(video, (resize_height, resize_width))

# num_points = 30  # @param {type: "integer"}
# query_points = sample_random_points(0, frames.shape[1], frames.shape[2], num_points)

# Load points
select_points = np.load('selected_points.npy')
query_points = convert_select_points_to_query_points(0, select_points)
query_points = transforms.convert_grid_coordinates(
    query_points, (1, height, width), (1, resize_height, resize_width), coordinate_format='tyx')
tracks, visibles = inference(frames, query_points)

# Visualize sparse point tracks
tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
video_viz = viz_utils.paint_point_track(video, tracks, visibles)
media.write_video('tapnet_sample_result.mp4', video_viz, fps=10)