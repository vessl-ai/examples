import argparse
import PIL.Image as Image

import gradio as gr
from ultralytics import YOLO


def predict_image(img, conf_threshold, iou_threshold):
    """Predicts objects in an image using a YOLO11 model with adjustable confidence and IOU thresholds."""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im


def main(port: int):
    demo = gr.Interface(
        fn=predict_image,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
            gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
        ],
        outputs=gr.Image(type="pil", label="Result"),
        title="Ultralytics Gradio",
        description="Upload images for inference.",
    )
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="YOLO Object Detector",
        description="Detect objects with Ultralytics YOLO model.",
    )
    parser.add_argument("model", nargs="?", default="yolo11n.pt", type=str, help="YOLO model checkpoint path.")
    parser.add_argument("--port", default=7860, type=int, help="Port to run the Gradio server.")
    args = parser.parse_args()

    model = YOLO(args.model)

    main(args.port)
