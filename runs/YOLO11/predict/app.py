import argparse
import os
import PIL.Image as Image
from time import sleep

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


def close_app():
    gr.Info("Terminated the app!")
    sleep(1)
    os._exit(0)


def main(port: int):
    with gr.Blocks(title="Ultralytics YOLO Gradio") as demo:
        with gr.Row():
            gr.Markdown(
                """<h2>Ultralytics YOLO</h2>
                <h3>Upload an image for inference with Ultralytics YOLO.</h3>"""
            )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    input_image = gr.Image(type="pil", label="Upload Image")
                with gr.Row():
                    with gr.Accordion("Advanced options", open=True):
                        confidence_slider = gr.Slider(
                            minimum=0, maximum=1, value=0.25, label="Confidence threshold"
                        )
                        iou_slider = gr.Slider(
                            minimum=0, maximum=1, value=0.45, label="IoU threshold"
                        )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                with gr.Row():
                    output_image = gr.Image(type="pil", label="result")
        with gr.Row():
            close_button = gr.Button("Close the app", variant="stop")
        
        submit_btn.click(
            predict_image,
            inputs=[input_image, confidence_slider, iou_slider],
            outputs=[output_image],
        )
        close_button.click(fn=lambda: gr.update(interactive=False), outputs=[close_button]).then(fn=close_app)

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
