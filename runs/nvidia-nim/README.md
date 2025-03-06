## NVIDIA NIM
This repository contains a VESSL RUN Template to run Meta's `Llama-3.1-8B-Instruct` model with self-hosted NVIDIA NIM.

### Instruction
1. Generate an NGC API key if you don't have one. Please refer to the [NVIDIA Documentation](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-personal-api-key) for the detailed guide.

2. Add a Docker credential at the **Integrations** tab of your organization settings page:
    - Credential name: `ngc-credential` 
    - ID: `$oauthtoken`
    - Password: enter the NGC API key generated above.

3. Add a secret at the **secrets** tab of your organization settings page:
    - Secret name: `ngc-api-key`
    - Secret value: enter the NGC API key generated above.

4. Install VESSL CLI and configure your identity:
    ```bash
    $ pip install --upgrade vessl
    $ vessl configure
    ```

5. (Optional) Create a new project (replace `${PROJECT_NAME}` with the project name):
    ```bash
    $ vessl project create ${PROJECT_NAME}
    $ vessl configure -p ${PROJECT_NAME}
    ```

6. Clone the repository:
    ```bash
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/runs/nvidia-nim
    ```

7. Deploy the application with the provided YAML file ([`run.yaml`](./run.yaml)):
    ```bash
    $ vessl run create -f run.yaml
    ```

For additional information and support, please refer to the [VESSL documentation](https://docs.vessl.ai).