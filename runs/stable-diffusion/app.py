import os

import torch
from diffusers import StableDiffusionPipeline

device = torch.device("cpu")
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
print("RUNNING ON:", device)

pipe = StableDiffusionPipeline.from_pretrained(os.environ.get("MODEL_NAME"))
pipe = pipe.to(device)

import random
import numpy as np
import gradio as gr

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1536


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@torch.inference_mode()
def generate(
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    randomize_seed: bool = True,
    width: int = 768,
    height: int = 768,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    num_images_per_prompt: int = 2,
):
    """Generate images using Stable Diffusion."""
    seed = randomize_seed_fn(seed, randomize_seed)
    print("seed:", seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_images_per_prompt,
    )

    return output.images


examples = [
    "An astronaut riding a green horse",
    "A mecha robot in a favela by Tarsila do Amaral",
    "The sprirt of a Tamagotchi wandering in the city of Los Angeles",
    "A delicious feijoada ramen dish",
]

with gr.Blocks(css="gradio_app/style.css") as demo:
    with gr.Column():
        prompt = gr.Text(
            label="Prompt",
            show_label=False,
            placeholder="Enter your prompt",
        )
        run_button = gr.Button("Run")
        with gr.Accordion("Advanced options", open=False):
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a Negative Prompt",
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            width = gr.Slider(
                label="Width",
                minimum=512,
                maximum=MAX_IMAGE_SIZE,
                step=128,
                value=768,
            )
            height = gr.Slider(
                label="Height",
                minimum=512,
                maximum=MAX_IMAGE_SIZE,
                step=128,
                value=768,
            )
            num_images_per_prompt = gr.Slider(
                label="Number of Images",
                minimum=1,
                maximum=2,
                step=1,
                value=2,
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0,
                maximum=20,
                step=0.1,
                value=7.5,
            )
            num_inference_steps = gr.Slider(
                label="Inference Steps",
                minimum=10,
                maximum=100,
                step=1,
                value=50,
            )

    with gr.Column():
        result = gr.Gallery(label="Result", show_label=False)

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=result,
        fn=generate,
    )

    inputs = [
        prompt,
        negative_prompt,
        seed,
        randomize_seed,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt,
    ]
    prompt.submit(
        fn=generate,
        inputs=inputs,
        outputs=result,
    )
    negative_prompt.submit(
        fn=generate,
        inputs=inputs,
        outputs=result,
    )
    run_button.click(
        fn=generate,
        inputs=inputs,
        outputs=result,
    )

demo.queue(20).launch(server_name="0.0.0.0")
