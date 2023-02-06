import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image

## 제목은 hugging face Rola finetuning with vessl ##


## 타인은 지옥이다로 finetuning을 한다던가 그런걸로 해야겠다.

## prompt
if __name__ == "__main__"  :

    # model_id_or_path = "runwayml/stable-diffusion-v1-5"
    #
    # pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    # pipe = pipe.to("cuda")
    # print("number of parameter vae", sum([p.numel() for p in pipe.vae.parameters() if p.requires_grad]))
    # print("number of parameter text_enconder", sum([p.numel() for p in pipe.text_encoder.parameters() if p.requires_grad]))
    # print("number of parameter unet", sum([p.numel() for p in pipe.unet.parameters() if p.requires_grad]))
    # prompt = "a photo of an woman riding a horse on ocean"
    # generated_images = pipe(prompt).images
    # print(len(generated_images))

    generated_images = Image.open('test2.png')

    plt.imshow(generated_images)

    plt.show()