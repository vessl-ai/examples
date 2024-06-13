# Instance Diffusion

This repository contains code to run Streamlit app for [Instance Diffusion](https://github.com/frank-xwang/InstanceDiffusion) that adds precise instance-level control to text-to-image diffusion models.

InstanceDiffusion supports free-form language conditions per instance and allows flexible ways to specify instance locations such as simple single points, scribbles, bounding boxes or intricate instance segmentation masks, and combinations thereof. Compared to the previous SOTA, InstanceDiffusion achieves 2.0 times higher AP50 for box inputs and 1.7 times higher IoU for mask inputs.

## Running Locally

1. Clone the repository:
   ```bash
   $ git clone https://github.com/vessl-ai/examples.git
   $ cd examples/instance-diffusion
   ```
2. Install required dependencies:
   ```bash
   $ pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   $ streamlit run app.py
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
   $ cd examples/instance-diffusion
   ```
4. Deploy the application with the provided YAML file ([`run.yaml`](./run.yaml)):
   ```bash
   $ vessl run create -f run.yaml
   ```

For additional information and support, please refer to the [VESSL documentation](https://docs.vessl.ai).

## Citation

```
@misc{wang2024instancediffusion,
      title={InstanceDiffusion: Instance-level Control for Image Generation},
      author={Xudong Wang and Trevor Darrell and Sai Saketh Rambhatla and Rohit Girdhar and Ishan Misra},
      year={2024},
      eprint={2402.03290},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
