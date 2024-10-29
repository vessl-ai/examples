# Deploy Llama 3.1 70B Model with VESSL Service
## Instruction
### Clone Git Repository
```sh
$ git clone https://github.com/vessl-ai/examples.git
$ cd examples/services/llama-3.1-70b
```
### Create Service Revision
```sh
$ vessl service create -f service.yaml
```
### Activate Revision (if it is not running)
```sh
$ vessl service split-traffic --service ${service name}
```
