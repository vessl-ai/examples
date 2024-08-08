# Deploy Llama 3 8B Model with VESSL Serve
## Instruction
### Clone Git Repository
```sh
$ git clone https://github.com/vessl-ai/examples.git
$ cd examples/serve-llama-3
```
### Create Service Revision
```sh
$ vessl service create -f serve.yaml
```
### Activate Revision (if it is not running)
```sh
$ vessl serve update --service llama-3-chatbot --interactive
```
