import argparse
import torch
import numpy as np
import uvicorn

from fastapi import FastAPI, UploadFile
from PIL import Image
from io import BytesIO

from model import Net

app = FastAPI()


def load_model(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_image(data):
    return Image.open(BytesIO(data))


@app.post("/predict_image/")
async def predict(file: UploadFile):
    image = load_image(await file.read())
    pix = np.array(image)
    pix = pix / 255.0
    data = pix.reshape(-1, 28, 28)
    infer_torch_data = torch.utils.data.TensorDataset(torch.from_numpy(data).unsqueeze(1))
    infer_dataloader = torch.utils.data.DataLoader(infer_torch_data, batch_size=128, shuffle=False)
    for data in infer_dataloader:
        infer_data = data[0].float()

    with torch.no_grad():
        result = model(infer_data).argmax(dim=1, keepdim=False)

    return {"result": result.item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path')
    args = parser.parse_args()

    model = load_model(args.model_path)

    uvicorn.run(app, host="0.0.0.0", port=5000)
