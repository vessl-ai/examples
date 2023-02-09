# Generative model

## Text_to_image fine-tuning by LoRa on Bayc text-image dataset

### Dataset 
Image-Text pair with auto generated

You can train with your own dataset using huggingface dataset or Vessl dataset mount
- Check dataset format url below
- Also, you can modify data pre-processing code in LoRa.py 

https://huggingface.co/datasets/VESSL/Bored_Ape_NFT_text


### Reference code 
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

### Prerequisite

#### You need to install diffusers>=0.13.0.dev0
git clone to install dev version of diffusers (may change in future)
```bash
pip install git+https://github.com/huggingface/diffusers.git
```

### Start Command
```bash
install -r requirements.txt 

accelerate launch --mixed_precision="bf16" --multi_gpu \
examples/generative_model/LoRa.py --random_flip --on_vessl=True  \
--train_batch_size=${train_batch_size} --num_train_epochs=$(num_train_epochs)  \
--learning_rate=${learning_rate} --seed=${seed} \
--output_dir="/output" --validation_prompt=${validation_prompt} 

```


> Noted that you should add [hyperparameters](../README.md) as env variables to the start command

### Hyperparameters
  ```bash
    num_train_epochs # Number of training epoch [default: 1]
    seed # random seed
    validation_prompt # prompt for validation starts with "An ape with" [ex "An ape with red hair and bored eyes"] 
    train_batch_size # batch_size on each process [default: 1] 
    learning_rate # training learning rate [default: 5e-5]
   ```

### VESSL

You can try it on vessl more easily! Visit https://vessl.ai/ for easy and reproducible training and validation.
