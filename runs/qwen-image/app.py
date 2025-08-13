import os
import uuid

import gradio as gr
import torch
from diffusers import DiffusionPipeline

MODEL_NAME = "Qwen/Qwen-Image"


class QwenImageGenerator:
    def __init__(self):
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            self.device = "cuda"
        else:
            torch_dtype = torch.float32
            self.device = "cpu"

        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
        )
        self.pipe.to(self.device)

    @torch.inference_mode()
    def generate_image(
        self,
        width,
        height,
        num_steps,
        seed,
        prompt,
    ):
        seed = int(seed)
        if seed == -1:
            seed = None

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)

        positive_magic = "Ultra HD, 4K, cinematic composition."
        negative_prompt = " "

        image = self.pipe(
            prompt=prompt + positive_magic,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        filename = f"output/gradio/{uuid.uuid4()}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        image.save(filename, format="png")

        return image, filename


def create_demo():
    generator = QwenImageGenerator()

    with gr.Blocks() as demo:
        gr.Markdown("# Qwen-Image Image Generation Demo")

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
                seed,
                prompt,
            ],
            outputs=[output_image, download_btn],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0")
