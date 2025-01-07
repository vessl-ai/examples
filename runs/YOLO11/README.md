# YOLO11
This repository contains code to run a Gradio app with [YOLO11](https://yolo11.com), the latest iteration in the Ultralytics YOLO series.

YOLO11 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLO11 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.

## Running Locally
1. Clone the repository:
    ```bash
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/runs/YOLO11
    ```
2. Install required dependencies:
    ```bash
    $ pip install gradio ultralytics
    ```
3. Run the application:
    ```bash
    $ python app.py
    ```
4. Access the Gradio interface by opening http://localhost:7860 in your web browser.


## Deploying with VESSL Run
VESSL is a platform for deploying and managing AI applications. It allows you to deploy your AI applications on the cloud with a single command, and provides a web interface for managing your applications.

To deploy the YOLO11 Gradio app, follow the steps below:

1. Install VESSL CLI and configure your identity:
    ```sh
    $ pip install --upgrade vessl
    $ vessl configure
    ```
2. (Optional) Create a new project (replace `${PROJECT_NAME}` with the name you want):
    ```sh
    $ vessl project create ${PROJECT_NAME}
    $ vessl configure -p ${PROJECT_NAME}
    ```
3. Clone this repository:
    ```sh
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/runs/YOLO11
    ```
4. Deploy the application with the provided YAML file ([`run.yaml`](./run.yaml)):
    ```sh
    $ vessl run create -f run.yaml
    ```

For additional information and support, please refer to the [VESSL documentation](https://docs.vessl.ai).