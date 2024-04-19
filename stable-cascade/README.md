# Stable Cascade
![img](https://github.com/Stability-AI/StableCascade/blob/master/figures/collage_1.jpg)

This repository contains code to run a Gradio app with [Stable Cascade](https://github.com/Stability-AI/StableCascade), a text to image model build upon the Würstchen architecture.

[Würstchen](https://openreview.net/forum?id=gU58d5QeGv) is a novel architecture for text-to-image synthesis that combines competitive performance with unprecedented cost-effectiveness for large-scale text-to-image diffusion models. It leverages a detailed but extremely compact semantic image representation to guide the diffusion process, which provides much more detailed guidance compared to latent representations of language and significantly reduces the computational requirements to achieve state-of-the-art results.

## Running Locally
1. Clone the repository:
    ```bash
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/stable-cascade
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

To deploy the Stable Cascade Gradio app, follow the steps below:

1. Create a new project on VESSL.
2. Deploy the application with the provided YAML file ([`run.yaml`](./run.yaml)):
    ```bash
    $ vessl run create -f run.yaml
    ```

For additional information and support, please refer to the [VESSL documentation](https://docs.vessl.ai).

## Citation
```bibtex
@misc{pernias2023wuerstchen,
      title={Wuerstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models}, 
      author={Pablo Pernias and Dominic Rampas and Mats L. Richter and Christopher J. Pal and Marc Aubreville},
      year={2023},
      eprint={2306.00637},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```