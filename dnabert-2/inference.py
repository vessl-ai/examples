import os

import gradio as gr
import torch
import transformers

model_path = os.getenv("MODEL_PATH", "/model")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    model_path,
    trust_remote_code=True,
)

def infer(dna: str) -> int:
    inputs = tokenizer(dna, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        outputs = model(inputs).logits
    predicted_class = torch.argmax(outputs[0]).item()
    return predicted_class

iface = gr.Interface(fn=infer, inputs='text', outputs='text')
iface.launch(server_name="0.0.0.0")