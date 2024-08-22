import os
import uuid

import gradio as gr
import torch
from diffusers import FluxPipeline

MODEL_NAME = "black-forest-labs/FLUX.1-schnell"


class FluxGenenrator:
    def __init__(self, offload: bool):
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        if offload:
            self.pipe.enable_sequential_cpu_offload()  # offloads modules to CPU on a submodule level (rather than model level)

    @torch.inference_mode()
    def generate_image(
        self,
        width,
        height,
        num_steps,
        guidance,
        seed,
        prompt,
    ):
        seed = int(seed)
        if seed == -1:
            seed = None

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator = generator.manual_seed(seed)

        image = self.pipe(
            prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        filename = f"output/gradio/{uuid.uuid4()}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        image.save(filename, format="png")

        return image, filename


def create_demo(offload: bool):
    generator = FluxGenenrator(offload)

    with gr.Blocks() as demo:
        gr.Markdown("# Flux.1-Schnell Image Generation Demo")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    value='A photo of the space and nebulae. An astronaut is floating in the middle and holding a huge taco. The word "VESSL AI" is written in a bottom right corner, in blue stroke.',
                )

                with gr.Accordion("Advanced Options", open=False):
                    width = gr.Slider(128, 8192, 1360, step=16, label="Width")
                    height = gr.Slider(128, 8192, 768, step=16, label="Height")
                    num_steps = gr.Slider(1, 50, 4, step=1, label="Number of steps")
                    guidance = gr.Slider(
                        1.0,
                        10.0,
                        3.5,
                        step=0.1,
                        label="Guidance",
                        interactive=False,
                    )
                    seed = gr.Textbox(-1, label="Seed (-1 for random)")

                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                download_btn = gr.File(label="Download full-resolution")

        generate_btn.click(
            fn=generator.generate_image,
            inputs=[
                width,
                height,
                num_steps,
                guidance,
                seed,
                prompt,
            ],
            outputs=[output_image, download_btn],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument(
        "--offload", action="store_true", help="Offload model to CPU when not in use"
    )
    args = parser.parse_args()

    demo = create_demo(args.offload)
    demo.launch(server_name="0.0.0.0")
