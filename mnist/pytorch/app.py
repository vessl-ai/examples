import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import Net  # model.py에서 Net 모델을 가져옵니다.
import numpy as np

# 모델 로드 함수
def load_model(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 이미지 전처리 함수
def preprocess(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(image).unsqueeze(0)

# 예측 함수
def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        return outputs.argmax(dim=1, keepdim=True).item()

# Streamlit 앱
def main():
    st.title("MNIST Digit Recognizer")
    model_path = "/root/model/model.pt"  # 모델 파일 경로. 필요에 따라 경로를 수정하세요.
    model = load_model(model_path)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        image = preprocess(image)
        prediction = predict(model, image)
        st.write(f"Predicted Digit: {prediction}")

if __name__ == "__main__":
    main()
