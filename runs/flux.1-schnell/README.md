# FLUX.1-schnell

![FLUX.1-schnell](./assets/flux-1-schnell-examples.jpeg)

FLUX.1 suite is a set of text-to-image models that define a new state-of-the-art in image detail, prompt adherence, style diversity and scene complexity for text-to-image synthesis, developed by [Black Forest Labs](https://blackforestlabs.ai).

FLUX.1 [schnell] is the fastest model among FLUX.1 family, which is a 12 billion parameter rectified flow transformer. Trained using latent adversarial diffusion distillation, FLUX.1 [schnell] can generate high-quality images in only 1 to 4 steps. Also, the code and weight is released under the `apache-2.0` licence, so that the individual can generate the high quality with FLUX.1 [schnell] in any purpose.

## Running Locally
***Note:** You have to use a GPU machine with CUDA to run FLUX.1 model.*

1. Prepare environment. Using virtual environment is strongly recommended.

    ```sh
    python -m venv .venv

    source .venv/bin/activate
    pip install -r requirements.txt
    ```

1. Launch the app. If the VRAM is not sufficient, add the `--offload` flag.

    ```sh
    python app.py --offload
    ```

1. Access the Gradio interface by opening http://localhost:7860 in your web browser.

## Deploying with VESSL Run
VESSL is a platform for deploying and managing AI applications. It allows you to deploy your AI applications on the cloud with a single command, and provides a web interface for managing your applications.

To deploy the FLUX.1 [schnell] Gradio app, follow the steps below:

1. Install VESSL CLI and configure your identity:
    ```sh
    $ pip install --upgrade vessl
    $ vessl configure
    ```

1. (Optional) Create a new project (replace `${PROJECT_NAME}` with the name you want):
    ```sh
    $ vessl project create ${PROJECT_NAME}
    $ vessl configure -p ${PROJECT_NAME}
    ```

1. Clone this repository:
    ```sh
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/runs/flux.1-schnell
    ```

1. Deploy the application with the provided YAML file ([`run.yaml`](./run.yaml)):
    ```sh
    $ vessl run create -f run.yaml
    ```

For additional information and support, please refer to the [VESSL documentation](https://docs.vessl.ai).
