# Production Serving with OpenLLM and VESSL Serve

[![English](https://img.shields.io/badge/language-EN-green)](README.md) [![Korean](https://img.shields.io/badge/language-한글-green)](README-ko.md)

> This document is currently in development. Please check back later for updates.

## Deploying Revision

```sh
# Deploy revision for original model
vessl serve revision create --serving openllm -f openllm-serve.yaml

# Deploy revision for fine-tuned model
vessl serve revision create --serving openllm -f openllm-serve-zephyr.yaml
```

## Check Revision Status

```sh
$ vessl serve revision list --serving openllm

  Number 8
  Status running
  Message OpenLLM mistralai/Mistral-7B-Instruct-v0.2 on vLLM

  Number 11
  Status running
  Message OpenLLM HuggingFaceH4/zephyr-7b-beta on vLLM
```

## Check Endpoint(Gateway) Configuration

```sh
$ vessl serve gateway show --serving openllm

  Enabled True
  Status success
  Endpoint model-service-gateway-xxxxxxxxxx.region.aws-cluster.vessl.ai
  Ingress Class nginx
  Annotations (empty)
  Traffic Targets
  - ########## 99 %:   8 (port 3000)
  - #          1  %:  11 (port 3000)
```

## Update Endpoint(Gateway) Configuration

```yaml
# gateway.yaml
enabled: true
targets:
  - number: 8
    port: 3000
    weight: 10
  - number: 11
    port: 3000
    weight: 90
```

```sh
vessl serve gateway update --serving openllm -f gateway.yaml
```
