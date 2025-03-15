## NVIDIA NeMo Curator
This repository contains a VESSL Run template to generate synthetic data using [NVIDIA NeMo Curator](https://github.com/NVIDIA/NeMo-Curator).

**NeMo Curator** is a Python library specifically designed for fast and scalable dataset preparation and curation for generative AI use cases such as foundation language model pretraining, text-to-image model training, domain-adaptive pretraining (DAPT), supervised fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT).

### Instruction
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
    $ cd examples/runs/nvidia-nemo-curator
    ```

4. Modify environment variables of the provided YAML file ([`run.yaml`](./run.yaml)) with the LLM model and endpoint you want to use:
    > ***NOTE:** It could cost you a lot of credits if you use paid LLM APIs. Consider deploying open source LLMs using [VESSL Service](https://docs.vessl.ai/guides/service/overview).* 
    ```yaml
    # run.yaml
    ...
    env:
        LLM_ENDPOINT: https://api.openai.com
        MODEL_NAME: gpt-4o-mini
        OPENAI_API_KEY: abc123
    ```

5. Create a new run with the modified YAML file:
    ```bash
    $ vessl run create -f run.yaml
    ```

For additional information and support, please refer to the [VESSL documentation](https://docs.vessl.ai).