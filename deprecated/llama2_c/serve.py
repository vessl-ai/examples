import uvicorn
import subprocess

from fastapi import FastAPI, Body
from pydantic import BaseModel


app = FastAPI()

model_path = "/root/examples/llama2_c/stories42M.bin"
model = None


def prepare():
    subprocess.run(f"cd ./llama2_c && gcc -O3 -o run run.c -lm && chmod u+x {model_path}", shell=True)


class PromptModel(BaseModel):
    prompts: str


@app.post("/generate/")
async def generate(p_model: PromptModel = Body(...)):
    command = f"cd ./llama2_c && ./run {model_path} -t 0.9 -n 256 -i \"{p_model.prompts}\""
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')


if __name__ == "__main__":
    prepare()
    uvicorn.run(app, host="0.0.0.0", port=5000)
