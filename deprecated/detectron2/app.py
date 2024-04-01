import argparse
import os
import time

import cv2
import streamlit as st

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


@st.cache()
def load_model(path, threshold):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold / 100
    cfg.MODEL.WEIGHTS = path
    predictor = DefaultPredictor(cfg)
    return predictor


def elapsed_time(fn, *args):
    start = time.time()
    output = fn(*args)
    end = time.time()

    elapsed = f"{end - start:.2f}"

    return elapsed, output


def inference(test_path, model_path):
    filenames = []
    for file in os.listdir(test_path):
        if file.endswith(".jpg") or file.endswith("jpeg") or file.endswith("png"):
            filenames.append(file)

    filenames.sort()
    img = st.sidebar.selectbox(
        "Select Image",
        filenames,
    )

    input_image = os.path.join(test_path, img)

    st.write("## Source image:")
    st.image(input_image, width=400)

    threshold = st.sidebar.slider("", 0, 100, 70)
    st.sidebar.write(f"### Threshold: {threshold}%")

    clicked = st.sidebar.button("Inference")
    if clicked:
        st.sidebar.success("Inference started!")

        elapsed, predictor = elapsed_time(load_model, model_path, threshold)
        st.write(f"## Elapsed time")
        st.write(f"- Load model: {elapsed} seconds")

        im = cv2.imread(input_image)
        elapsed, outputs = elapsed_time(predictor, im)
        st.write(f"- Inference: {elapsed} seconds")

        st.write("## Result image:")
        v = Visualizer(im[:, :, ::-1], scale=0.5, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        st.image(out.get_image()[:, :, ::-1], width=400)

        st.sidebar.success("Inference done!")


def read_file(path):
    return open(path).read()


def main(args):
    st.title("Inference Detectron2")

    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode", ["Instruction", "Inference", "Source code"]
    )
    if app_mode == "Instruction":
        st.write(
            "## A simple app to inference Detectron2\n"
            "### To inference with trained model:\n"
            "1. Click **Inference** app mode on the side bar\n"
            "2. Select the image\n"
            "3. Choose the threshold on the slider\n"
            "4. Click the **Inference** button\n"
            "5. The inference result will show up\n"
            "### To show the source code:\n"
            "1. Click **Source code** app mode on the side bar\n"
            "2. The source code will show up"
        )
    elif app_mode == "Inference":
        inference(args.test_path, args.model_path)
    elif app_mode == "Source code":
        st.code(read_file(args.app_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlit MNIST Demo")
    parser.add_argument(
        "--app-path",
        type=str,
        default="/work/examples/detectron2/app.py",
        help="source code path",
    )
    parser.add_argument(
        "--test-path", type=str, default="/input", help="test dataset path"
    )
    parser.add_argument(
        "--model-path", type=str, default="/input/model_final.pth", help="model path"
    )
    args = parser.parse_args()
    main(args)
