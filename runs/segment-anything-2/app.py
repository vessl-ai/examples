import gradio as gr
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


sam2_checkpoint = "/sam2-repo/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


with gr.Blocks() as demo:

    with gr.Row():
        input_img = gr.Image(label="Input")
        output_img = gr.Image(label="Segments")

    def segment_image(img):
        masks = mask_generator.generate(img)
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        segments = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
        for ann in sorted_anns:
            m = ann['segmentation']
            segments[m] = np.random.random(3)
        img = img.astype(np.float64) / 255
        out = img * 0.3 + segments * 0.7
        return out

    input_img.upload(segment_image, [input_img], output_img)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")