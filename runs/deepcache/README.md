# DeepCache

This repository contains code to run Streamlit app for [Stable Diffusion XL base model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) using [DeepCache](https://github.com/horseee/DeepCache), a novel training-free and almost lossless paradigm that accelerates diffusion models from the perspective of model architecture.

DeepCache capitalizes on the inherent temporal redundancy observed in the sequential denoising steps of diffusion models, which caches and retrieves features across adjacent denoising stages, thereby curtailing redundant computations. DeepCache accelerates Stable Diffusion v1.5 by 2.3x with only a 0.05 decline in CLIP Score, and LDM-4-G(ImageNet) by 4.1x with a 0.22 decrease in FID.

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
@inproceedings{ma2023deepcache,
  title={DeepCache: Accelerating Diffusion Models for Free},
  author={Ma, Xinyin and Fang, Gongfan and Wang, Xinchao},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```