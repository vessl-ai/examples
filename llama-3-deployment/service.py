from typing import List

from pydantic import BaseModel
import torch
import transformers

import vessl


class InputMessage(BaseModel):
    role: str
    content: str

        
class InputType(BaseModel):
    messages: List[InputMessage]


class OutputType(BaseModel):
    generated_text: str

        
class Service(vessl.RunnerBase[InputType, OutputType]):
    @staticmethod
    def load_model(props, artifacts):
        pipeline = transformers.pipeline(
            "text-generation",
            model="casperhansen/llama-3-8b-instruct-awq",
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )
        return pipeline
    
    @staticmethod
    def preprocess_data(data):
        return data
    
    @staticmethod
    def predict(model, data):
        prompt = model.tokenizer.apply_chat_template(
                data, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            model.tokenizer.eos_token_id,
            model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        return outputs
    
    @staticmethod
    def postprocess_data(data):
        return data[0]