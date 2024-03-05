import argparse
import torch
import vessl

from vessl import Servo, Input

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
)

MODEL_PATH = "/model/"


class Llama2Serve(Servo):
    def setup(self):
        # Set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            cache_dir=MODEL_PATH,
            device_map="auto"
        )


        # Set Model
        self.model = LlamaForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            cache_dir=MODEL_PATH,
            device_map="auto"
        )

    def inference(self, prompt: str = Input(description="Llama2 prompt")) -> str:
        # Preprocess
        self.tokenizer.bos_token_id = 1
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=-0.3,
            top_p=0.75,
            top_k=40,
            num_beams=1,
        )

        # Predict
        with torch.no_grad():
            generation_outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=600,
            )

        # Postprocess
        s = generation_outputs.sequences[0]
        decoded_output = self.tokenizer.decode(s, skip_special_tokens=True).strip()

        return decoded_output

