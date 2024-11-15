# Stable Diffusion

The repository is for creating a sample app for Stable Diffusion with the HuggingFace Diffusers and Gradio.

## Running Locally
1. Clone the repository:
    ```bash
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/runs/stable-diffusion
    ```
2. Install required dependencies:
    ```bash
    $ pip install -r requirements.txt
    ```
3. Run the application:
    ```bash
    $ python app.py
    ```
4. Access the Gradio interface by opening http://localhost:7860 in your web browser.

## Deploying with VESSL Run
VESSL is a platform for deploying and managing AI applications. It allows you to deploy your AI applications on the cloud with a single command, and provides a web interface for managing your applications.

To deploy the Stable Diffusion Gradio app, follow the steps below:

1. Create a new project on VESSL.
2. Deploy the application with the provided YAML file ([`run.yaml`](./run.yaml)):
    ```bash
    $ vessl run create -f run.yaml
    ```

For additional information and support, please refer to the [VESSL documentation](https://docs.vessl.ai).