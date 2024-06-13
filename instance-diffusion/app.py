import argparse
import json
import os
import time
import uuid
from functools import partial
from tkinter.messagebox import NO

import numpy as np
import streamlit as st
import torch
from dataset.decode_item import (decodeToBinaryMask, reorder_scribbles,
                                 sample_random_points_from_mask,
                                 sample_sparse_points_from_mask)
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from inference import draw_boxes, infer_by_streamlit, rescale_box
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.plms_instance import PLMSSamplerInst
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from skimage.transform import resize
from utils.checkpoint import load_model_ckpt
from utils.input import convert_points, prepare_batch, prepare_instance_meta
from utils.model import (alpha_generator, create_clip_pretrain_model,
                         set_alpha_scale)

# --num_images 8   --output OUTPUT/   --input_json demos/demo_cat_dog_robin.json   --ckpt pretrained/instancediffusion_sd15.pth   --test_config configs/test_box.yaml   --guidance_scale 7.5   --alpha 0.8   --seed 0   --mis 0.36   --cascade_strength 0.4

st.title("Instance Diffusion")

        
with st.container(border=True):
    st.write("Root config")
    output="OUTPUT"
    num_images = st.number_input("num_images", value=8)
    guidance_scale = st.number_input("guidance_scale", value=7.5)
    negative_prompt = st.text_input("negative_prompt", value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
    alpha = st.number_input("alpha", value=0.8)
    seed = st.number_input("seed", value=0)
    mis = st.number_input("mis", value=0.36, max_value=0.4)
    cascade_strength = st.number_input("cascade_strenth", value=0.4)

    test_config = "configs/test_box.yaml"
    ckpt = "pretrained/instancediffusion_sd15.pth"

# ui_config, json_config = st.tabs(["Config by UI", "Config by json"])
# use_ui = True
    
# with ui_config:
with st.container(border=True):
    st.write("Main Config")
    caption = st.text_input("caption", value="a yellow American robin, brown Maltipoo dog, a gray British Shorthair in a stream, alongside with trees and rocks")
    width = st.number_input("width", value=512)
    height = st.number_input("height", value=512)

with st.container(border=True):
    sample_anno_bbox = [
        "0,51,179,230",
        "179,102,153,153",
        "332,102,179,255",
        "0,358,512,153"
    ]
    sample_anno_caption = [
        "a gray British Shorthair standing on a rock in the woods",
        "a yellow American robin standing on the rock",
        "a brown Maltipoo dog standing on the rock",
        "a close up of a small waterfall in the woods"
    ]
    annos = []
    num_annos = st.number_input("number of annotations", value=4)
    for i in range(num_annos):
        with st.container(border=True):
            st.write("annotation config #"+str(i+1))
            bbox_input = st.text_input("bounding box for #{}: xmin,ymin,width,height".format(i), value=sample_anno_bbox[i])
            bbox = [int(x) for x in bbox_input.replace(" ", "").split(",")] if len(bbox_input) > 0 else []
            point_input = st.text_input(f"point for #{i}: x,y")
            point = [int(p) for p in point_input.replace(" ", "").split(",")] if len(point_input) > 0 else []
            scribble_input = st.text_input(f"scribble for #{i}: x1,y1,...,x20,y20")
            scribble = [int(e) for e in scribble_input.replace(" ", "").split(",")] if len(scribble_input) > 0 else []
            anno_caption = st.text_input("caption for #{}".format(i), value=sample_anno_caption[i])
            category_name = st.text_input("category_name for #{}".format(i), value="")
            num_masks = st.number_input("number of masks for #{}".format(i), value=0)
            masks = []
            for j in range(num_masks):
                with st.container(border=True):
                    st.write("mask config #"+str(j+1))
                    mask_category = st.text_input("category for annotation-{} mask-{}".format(i, j))
                    mask_caption = st.text_input("caption for annotation-{} mask {}".format(i, j))
                    masks.append({"category": mask_category, "caption": mask_caption})
            annos.append({
                "bbox": bbox,
                "point": point,
                "scribble": scribble,
                "masks": masks,
                "category_name": category_name,
                "caption": anno_caption,
            })

input_config = {
    "caption": caption,
    "width": width,
    "height": height,
    "annos": annos
}
# use_ui = True
# with json_config:
#     with st.container(border=True):
#         json_input_raw = st.text_area("Paste your json input", value="""
# {
#   "caption": "a yellow American robin, brown Maltipoo dog, a gray British Shorthair in a stream, alongside with trees and rocks",
#   "width": 512,
#   "height": 512,
#   "annos": [
#     {
#       "bbox": [
#         0,
#         51,
#         179,
#         230
#       ],
#       "mask": [],
#       "category_name": "",
#       "caption": "a gray British Shorthair standing on a rock in the woods"
#     },
#     {
#       "bbox": [
#         179,
#         102,
#         153,
#         153
#       ],
#       "mask": [],
#       "category_name": "",
#       "caption": "a yellow American robin standing on the rock"
#     },
#     {
#       "bbox": [
#         332,
#         102,
#         179,
#         255
#       ],
#       "mask": [],
#       "category_name": "",
#       "caption": "a brown Maltipoo dog standing on the rock"
#     },
#     {
#       "bbox": [
#         0,
#         358,
#         512,
#         153
#       ],
#       "mask": [],
#       "category_name": "",
#       "caption": "a close up of a small waterfall in the woods"
#     }
#   ]
# }""")
        
#         json_input_config = json.loads(json_input_raw)
#         if "caption" not in json_input_config or "width" not in json_input_config or "height" not in json_input_config or "annos" not in json_input_config:
#             st.write("Wrong json format")
#         use_ui = False
        
            
if st.button("preview box"):
#     if use_ui:
#         input_config = ui_input_config
#     else:
#         input_config = json_input_config
    boxes = []
    phrases = []
    for inst_idx in range(len(input_config['annos'])):
        if "bbox" not in input_config['annos'][inst_idx]: 
                boxes.append([0,0,0,0])
        else:
            boxes.append(input_config['annos'][inst_idx]['bbox'])
        phrases.append(input_config['annos'][inst_idx]['caption'])
    locations = [rescale_box(box, input_config['width'], input_config['height']) for box in boxes]
    image_boxes = draw_boxes( locations, phrases, input_config['caption'] + ";alpha=" + str(alpha) )
    st.image(image_boxes, "preview")

if st.button("start"):
#     if use_ui:
#         input_config = ui_input_config
#     else:
#         input_config = json_input_config
    if not os.path.exists(output):
        os.makedirs(output)
    else:
        os.system(f"rm -rf {output}")
        os.makedirs(output)
    input_json = "{}/input.json".format(output)
    with open(input_json, "w") as f:
        f.write(json.dumps(input_config))
    generation_key = str(uuid.uuid4())
    output = f"{output}/{generation_key}"
    with st.spinner("Generating..."):
        infer_by_streamlit(
            output=output,
            num_images=num_images,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            input_json=input_json,
            ckpt=ckpt,
            seed=seed,
            alpha=alpha,
            mis=mis,
            cascade_strength=cascade_strength,
            test_config=test_config
        )

        images = {}
        for i in range(num_images):
            box = f"{output}/output_images/{i}_boxes.png"
            if os.path.exists(box):
                images[f"box{i}"] = Image.open(box)
                st.image(images[f"box{i}"], "boxes")

            sample = f"{output}/output_images/{i}.png"
            if os.path.exists(sample):
                images[f"sample{i}"] = Image.open(sample)
                st.image(images[f"sample{i}"], f"{i}")

            refined = f"{output}/output_images/{i}_xl_s{cascade_strength}_n{20}.png"
            if os.path.exists(refined):
                images[f"refined{i}"] = Image.open(refined)
                st.image(images[f"refined{i}"], f"{i}-refined")
        