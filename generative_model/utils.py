import vessl
import os

from typing import Optional
from huggingface_hub import HfFolder, whoami

class VesslLogger:
    """VESSL logger"""
    def __init__(self, save_image=False):
        """Initializer"""
        self._log = {}

        if save_image:
            self.save_image = True
            self._examples= []
        else:
            self.save_image = False

    def log(self, step, metric, value):
        """Log metrics. Each metric's log will be stored in the corresponding list.
        Args:
            step (int): step
            metric (str): Metric name.
            value (float): Value.
        """
        if metric not in self._log:
            self._log[metric] = []
        self._log[metric].append(value)
        vessl.log(step=step, payload={
            metric: value,
        })

    def get_log(self):
        """Getter
        Returns:
            dict: Log metrics.
        """
        return self._log

    def save_images(self, image, caption):
        self._examples.append(vessl.Image(image, caption))

    def log_images(self, title="Examples"):
        vessl.log({title :
                self._examples})


def save_model_card(repo_name, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_name}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


DATASET_NAME_MAPPING = {
    "VESSL/Bored_Ape_NFT" : ("image", "text"),
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

def print_num_param(model, name) :
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters in {name} is {num_param}")
