# Deploy Llama 3 model with VESSL Serve
## Instruction
1. Install VESSL CLI and configure your identity.
    ```sh
    $ pip install --upgrade vessl
    $ vessl configure
    ```
2. Clone this repository.
    ```sh
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/llama-3-deployment
    ```

3. Create a model repository.
    ```sh
    $ vessl model-repository create llama-3-deployment

    Organization: demo
    Created 'llama-3-deployment'.
    For more info: https://vessl.ai/demo/models/llama-3-deployment
    ```

4. Register model to the model repository.
    ```sh
    $ vessl model register

    Organization: demo
    Type of model to register (vessl, bento, hf-transformers, hf-diffusers) [vessl]: vessl
    [?] Model repository: 
    > llama-3-deployment

    [?] Generating entrypoint as `vessl model launch service.py:Service -p 3000`. Proceed? (No to input manually) (Y/n): y

    [?] Python version? (detected: 3.10): 3.10
    [?] Framework type: 
    > torch
      tensorflow

    [?] PyTorch version? (contact support@vessl.ai if you cannot find expected.): 
    > 2.3.0
      2.2.0
      1.14.0

    [?] CUDA version?: 
    > 12.4
      12.3

    [?] Path to requirements file? [detected: requirements.txt] (optional, press Enter to skip): requirements.txt
    [?] Model number (optional): 
    Generated ignore file.
    [?] Register and upload current directory as model? (Y/n): y

    Creating a new model.
    Created a new model with number 1.
    Lockfile saved.
    llama-3-deployment-1 /code/examples/llama-3-deployment /
    Uploading 4 file(s) (768.0B)...
    Total 4 file(s) uploaded.
    Registered llama-3-deplyment-1.
    ```

5. Create a service.
    ![click 'New service' on the service panel](assets/service-creation-1.png)
    ![create new service](assets/service-creation-2.png)

6. Create a YAML file to configure the service revision. Replace `${API key value}` with your own API key.
    ```sh
    $ vessl serve create-yaml llama-3-textgen llama-3-deplyment-test --api-key

    Service name of llama-3-textgen found.
    Using vessl-gcp-oregon cluster configured by the service.
    [?] Preset: 
      cpu-small
      cpu-small-spot
      gpu-l4-small
      gpu-l4-small-spot
      gpu-l4-medium
      gpu-l4-medium-spot
      gpu-l4-large
      gpu-l4-large-spot
      gpu-v100-small
    > gpu-v100-small-spot

    Select API key for authentication.
    [?] Secret:
    > Create a new secret

    [?] Secret name: llama-3-api-key
    [?] Secret value: ${API key value}
    Secret llama-3-api-key created.
    service.yaml created.
    ```

7. Create a service revision and deploy.
    ```sh
    $ vessl serve create -f service.yaml -a
    ```