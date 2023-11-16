from diffusers import StableDiffusionXLPipeline
import torch
import configparser
import sys, getopt

def main(argv):
    file_name = argv[0]
    prompt = "An astronaut riding a green horse" # Your prompt here
    neg_prompt = "ugly, blurry, poor quality, scary" # Negative prompt here

    try:
        opts, etc_args = getopt.getopt(argv[1:], \
                                 "hi:c:", ["help","prompt=","neg_prompt="])

    except getopt.GetoptError:
        print(file_name, '-p = <prompt name> -n = <neg_prompt name>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(file_name, '-p = <prompt name> -n = <neg_prompt name>')
            sys.exit()

        elif opt in ("-p", "--prompt"):
            prompt = arg

        elif opt in ("-n", "--neg_prompt"):
            neg_prompt = arg

    '''
    # Mandatory argument setting
    if len(prompt) < 1:
        print(file_name, "-p prompt is mandatory")
        sys.exit(2)

    print("prompt:", prompt)
    print("neg_prompt:",  neg_prompt)
    '''

    pipe = StableDiffusionXLPipeline.from_pretrained("/data/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")

    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    #prompt = "An astronaut riding a green horse" # Your prompt here
    #neg_prompt = "ugly, blurry, poor quality, scary" # Negative prompt here
    image = pipe(prompt=prompt, negative_prompt=neg_prompt).images[0]
    image.save(prompt + '.jpg')

if __name__ == "__main__":
    main(sys.argv)
