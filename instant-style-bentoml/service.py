import bentoml
from PIL.Image import Image


@bentoml.service(traffic={"timeout": 300}, workers=1)
class InstantStyle:
    def __init__(self) -> None:
        import torch
        from diffusers import LCMScheduler, StableDiffusionXLPipeline

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(device)

        self.pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
        )
        self.pipe.enable_vae_tiling()

        # configure ip-adapter scales.
        scale = {
            "down": {"block_2": [0.0, 1.0]},
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        self.pipe.set_ip_adapter_scale(scale)

        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
        self.pipe.scheduler = LCMScheduler.from_config(
            self.lcm_txt2img.scheduler.config
        )

    @bentoml.api
    def image2image(
        self,
        style_image: Image,
        source_image: Image,
        prompt: str,
        negative_prompt: str = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 10,
    ) -> Image:
        image = self.pipe(
            prompt=prompt,
            ip_adapter_image=[style_image, source_image],
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]

        return image
