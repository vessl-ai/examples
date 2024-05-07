import cv2
import numpy as np
import streamlit as st
import torch
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis
from instant_id.pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)


@st.cache_resource
def load_model() -> StableDiffusionXLInstantIDPipeline:
    # prepare 'antelopev2' under ./models

    # prepare models under ./checkpoints
    face_adapter = "./checkpoints/ip-adapter.bin"
    controlnet_path = "./checkpoints/ControlNetModel"

    # load IdentityNet
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        "stablediffusionapi/albedobase-xl",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=False,
    )
    pipe.cuda()
    pipe.enable_model_cpu_offload()

    # load adapter
    pipe.load_ip_adapter_instantid(face_adapter)
    pipe.load_lora_weights(
        "artificialguybr/StickersRedmond", weight_name="StickersRedmond.safetensors"
    )

    return pipe


@st.cache_resource
def get_antelope_app():
    app = FaceAnalysis(
        name="antelopev2",
        root="./",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    return app


def run_pipe(
    pipe,
    app,
    image,
    prompt: str,
    negative_prompt: str = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    num_images_per_prompt: int = 1,
):
    # prepare face emb
    face_info = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    face_info = sorted(
        face_info,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],
    )[-1]  # only use the maximum face
    face_emb = face_info["embedding"]
    face_kps = draw_kps(image, face_info["kps"])

    pipe.set_ip_adapter_scale(0.8)

    # generate image
    images = pipe(
        prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=0.8,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
    ).images

    return images
