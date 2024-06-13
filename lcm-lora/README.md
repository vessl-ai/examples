# LCM-LoRA

This repository contains code to run Streamlit app for [Stable Diffusion XL base model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) using [LCM-LoRA](https://github.com/luosiallen/latent-consistency-model), a LoRA adapter which can be attached to any compatible diffusion model and enables fast generation of high-quality images.

## Running Locally
1. Clone the repository:
    ```bash
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/deepcache
    ```
2. Install required dependencies:
    ```bash
    $ pip install -r requirements.txt
    ```
3. Run the application:
    ```bash
    $ streamlit run main.py
    ```
4. Access the Streamlit interface by opening http://localhost:8501 in your web browser.

## Deploying with VESSL Run
VESSL is a platform for deploying and managing AI applications. It allows you to deploy your AI applications on the cloud with a single command, and provides a web interface for managing your applications.

To deploy the Streamlit application with VESSL Run, follow the steps below:

1. Install VESSL CLI and configure your identity:
    ```bash
    $ pip install --upgrade vessl
    $ vessl configure
    ```
2. (Optional) Create a new project (replace `${PROJECT_NAME}` with the project name):
    ```bash
    $ vessl project create ${PROJECT_NAME}
    $ vessl configure -p ${PROJECT_NAME}
    ```
3. Clone the repository:
    ```bash
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/deepcache
    ```
4. Deploy the application with the provided YAML file ([`run.yaml`](./run.yaml)):
    ```bash
    $ vessl run create -f run.yaml
    ```

For additional information and support, please refer to the [VESSL documentation](https://docs.vessl.ai).

## Citation
```bibtex
@misc{luo2023latent,
  title={Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference},
  author={Simian Luo and Yiqin Tan and Longbo Huang and Jian Li and Hang Zhao},
  year={2023},
  eprint={2310.04378},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
