# YOLO11
This repository contains code to run a sample Gradio app with [YOLO11](https://yolo11.com), the latest iteration in the Ultralytics YOLO series, and train it.

YOLO11 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLO11 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.

## Running Locally
### Prerequisite
1. Clone the repository:
    ```bash
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/runs/YOLO11
    ```

### Running Gradio App
1. Install required dependencies:
    ```bash
    $ pip install gradio ultralytics
    ```
2. Run the application:
    ```bash
    $ python predict/app.py
    ```
3. Access the Gradio interface by opening http://localhost:7860 in your web browser.

### Training
1. Install required dependencies:
    ```bash
    $ pip install ultralytics
    ```
2. Run the training script:
    ```bash
    $ python train/train.py --output-path ${OUTPUT_PATH} --run-name ${RUN_NAME}
    ```

3. You can run the Gradio app using the trained model:
    ```bash
    $ python predict/app.py ${OUTPUT_PATH}/${RUN_NAME}/weights/best.pt
    ```


## Deploying with VESSL Run
VESSL is a platform for deploying and managing AI applications. It allows you to deploy your AI applications on the cloud with a single command, and provides a web interface for managing your applications.

### Prerequisites

1. Install VESSL CLI and configure your identity:
    ```bash
    $ pip install --upgrade vessl
    $ vessl configure
    ```
2. (Optional) Create a new project (replace `${PROJECT_NAME}` with the name you want):
    ```bash
    $ vessl project create ${PROJECT_NAME}
    $ vessl configure -p ${PROJECT_NAME}
    ```
3. Clone this repository:
    ```bash
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/runs/YOLO11
    ```

### Running Gradio App
1. Deploy the application with the provided YAML file ([`predict/run.yaml`](./predict/run.yaml)):
    ```bash
    $ vessl run create -f predict/run.yaml
    ```

### Training YOLO Model
1. Launch the training VESSL Run job with the provided yaml file ([`train/run.yaml`](./train/run.yaml)):
    ```bash
    $ vessl run create -f train/run.yaml
    ```

For additional information and support, please refer to the [VESSL documentation](https://docs.vessl.ai).

## Citation
```bibtex
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```