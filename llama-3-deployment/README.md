# Deploy Llama 3 model with VESSL Serve
This document provides step-by-step instructions for deploying the Llama 3 model using VESSL Serve.

## Prerequisites
Before you begin, ensure that you have the following:

- Git
- Python 3
- VESSL account

## Instruction
### 1. Install VESSL CLI and Configure Your Identity
First, install the VESSL CLI tool and configure it with your VESSL account credentials.
```sh
$ pip install --upgrade vessl
$ vessl configure
```
### 2. Clone the Repository
Next, clone the VESSL examples repository and navigate to this directory.
```sh
$ git clone https://github.com/vessl-ai/examples.git
$ cd examples/llama-3-deployment
```

### 3. Create a Model Repository
Create a new model repository in VESSL.
```sh
$ vessl model-repository create llama-3-deployment

Organization: demo
Created 'llama-3-deployment'.
For more info: https://vessl.ai/demo/models/llama-3-deployment
```

### 4. Register the Model
Register your model in the model repository you just created.
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

### 5. Create a Service
Create a new service in the VESSL platform.
1. Go to the VESSL web interface.
2. Click 'New service' on the service panel.
![click 'New service' on the service panel](assets/service-creation-1.png)
3. Follow the prompts to create a new service.
![create new service](assets/service-creation-2.png)

### 6. Create a YAML File to Configure the Service Revision
Create a YAML configuration file for the service revision. Replace `${API key value}` with your own API key.
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

### 7. Create a Service Revision and Deploy
Finally, create a service revision and deploy it with the YAML file you created above.
```sh
$ vessl serve create -f service.yaml -a
```
This will deploy the Llama 3 model using the specified configuration in the YAML file.

For more detailed information and troubleshooting, refer to the [VESSL documentation](https://docs.vessl.ai/) or contact [VESSL support](mailto:support@vessl.ai).