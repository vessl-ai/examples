## NVIDIA NIM
This repository contains a VESSL Service Template to deploy Meta's `Llama-3.1-8B-Instruct` model with self-hosted NVIDIA NIM.

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

5. Clone the repository:
    ```bash
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/runs/nvidia-nim
    ```

6. Create a new Service revision with the provided YAML file ([`service.yaml`](./service.yaml)):
    ```bash
    $ vessl service create -f service.yaml
    ```

For additional information and support, please refer to the [VESSL documentation](https://docs.vessl.ai).