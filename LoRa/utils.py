import os
from typing import Optional

import vessl
from huggingface_hub import HfFolder, whoami


class VesslLogger:
    """VESSL logger"""

    def __init__(self, save_image=False):
        """Initializer"""
        self._log = {}

        if save_image:
            self.save_image = True
            self._examples = []
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
        vessl.log(
            step=step,
            payload={
                metric: value,
            },
        )

    def get_log(self):
        """Getter
        Returns:
            dict: Log metrics.
        """
        return self._log

    def save_images(self, image, caption):
        self._examples.append(vessl.Image(image, caption))

    def log_images(self, title="Examples"):
        vessl.log({title: self._examples})


DATASET_NAME_MAPPING = {
    "VESSL/Bored_Ape_NFT_text": ("image", "text"),
}


def print_num_param(model, name):
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters in {name} is {num_param}")
