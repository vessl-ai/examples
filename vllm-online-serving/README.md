# Online Serving with vLLM and VESSL Run

[![English](https://img.shields.io/badge/language-EN-green)](README.md) [![Korean](https://img.shields.io/badge/language-한글-green)](README-ko.md)

This document outlines how to deploy a fast and efficient LLM API using VESSL Run and vLLM.
* [VESSL Run](https://run-docs.vessl.ai/) is a platform that enables easy building and deployment of AI models.
* [vLLM](https://vllm.ai/) is a library enhancing LLM inference performance, based on technologies like [PagedAttention](https://arxiv.org/pdf/2309.06180.pdf).

> **Note**:
> * Before starting, ensure you're logged into VESSL by running the following commands in the CLI environment:
>   ```sh
>   # Check if you are logged in
>   vessl whoami
> 
>   # If not, configure VESSL CLI with your VESSL credentials
>   vessl configure
>   ```
> * VESSL Run provides functionality to execute notebooks, training or inference services in a **single** container. For Production-level AI services that require features like Autoscaling or Load Balancing, please refer to [VESSL Serve](https://docs.vessl.ai/user-guide/serve).

## Deploying vLLM Server with VESSL Run

**Run** is the basic unit for task execution in VESSL. A Run's definition includes various details needed to execute a task such as code, commands, AI models, packages, environment variables, etc.


The definition of a Run is written in a YAML file. For instance, here is a snippet of the YAML file for this example:

```yaml
# vllm-online-serving.yaml
name: vllm-server
description: LLM server with vLLM and Prometheus monitoring
tags:
  - vllm
  - model=mistral-7b-instruct-v0.2
resources: # Resource requirements
  cluster: vessl-gcp-oregon
  preset: gpu-l4-small
image: quay.io/vessl-ai/torch:2.2.0-cuda12.3-r3 # Container image
import: # Code, data, or model to import
  /code/:
    git:
      url: github.com/vessl-ai/examples.git
      ref: main
run:
  - command: |- # Command to run the API server
      ...
    workdir: /code
ports: # Endpoint configuration
  - name: vllm
    type: http
    port: 8000
  - name: prometheus
    type: http
    port: 9090
```
This [vllm-online-serving.yaml](vllm-online-serving.yaml) file defines the following:
* The resources and container image to be used
* Git repository information for importing code
* Commands to run the monitoring tool Prometheus
* Commands to run a vLLM-based LLM API server
* Configuration to connect the API server and Prometheus endpoints

To create a Run, Use `vessl run` command with the YAML file.

```sh
vessl run create -f vllm-online-serving.yaml
```

Executing the above command will create a Run as shown in the screenshot below.

![](asset/run-demo.png)


> **Note**: For detailed instructions on creating a Run, please refer to the [VESSL Run Quickstart](https://run-docs.vessl.ai/docs/en/get-started/quickstart)!

## Accessing VESSL Run with Web Dashboard

After a Run is created, you can find the link to the Web Dashboard, where you can check the status of the Run, in the CLI environment as shown below.

![](asset/run-link.png)

The dashboard allows you to view detailed information about the Run, connected service endpoints, and easily replicate the Run with a single button click.

![](asset/run-details.png)

## Testing the API

Select Connect -> `vllm` in the Run Dashboard to navigate to the API endpoint. You can find the URL (`API_ENDPOINT_URL`) in the address bar of your web browser.

![API endpoint](asset/api-endpoint.png)

Navigate to `http://{API_ENDPOINT_URL}:8000/docs` to verify the API server is functioning correctly.

Send a simple HTTP POST request to test if the API server is operating correctly.

```sh
$ curl -X POST \
    http://{API_ENDPOINT_URL}/generate \
    -d '{"prompt":"What is the capital state of South Korea?"}'

{"text":["\n\nThe capital state of South Korea is Seoul.\n\n"]}
```

## Advanced: Benchmarking the API Server

`vLLM` provides the capability to collect key metrics necessary for monitoring LLM services based on Prometheus. Examples of metrics provided by the vLLM service include:
* E2E request latency: The time taken to send a request to the API server and receive a response


* Token throughput: The number of tokens processed/generated per second
* Time per first token: The time taken from receiving a request to generating the first token
* Cache utilization: The percentage of data stored in the GPU VRAM's KV cache that is utilized

> **Note**: The Metric functionality for LLM services is included in the latest upstream version of vLLM and has not been released as a Stable release as of February 2024. Please exercise caution when using it!

You can install vLLM and its required dependencies in your local environment, then run the benchmark script to evaluate the performance of the API server as follows.

```sh
# Install vLLM and required dependencies
pip install aiohttp vllm

# Clone the vLLM repository (which contains benchmark script)
git clone https://github.com/vllm-project/vllm

# Get sample prompt for benchmarking
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Run the benchmark script
python vllm/benchmarks/benchmark_serving.py \
  --backend vllm \
  --protocol_http \
  --host {API_SERVER_ENDPOINT} \
  --port 8000 \
  --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
  --request-rate 3 --dataset ShareGPT_V3_unfiltered_cleaned_split.json 
```

Prometheus is already running in the Run you initiated earlier. You can access Prometheus by selecting Connect -> `prometheus` in the Run Dashboard to view various metrics in the Prometheus UI and API server.

![](asset/vllm-prom.png)

You can also connect the Prometheus source to Grafana to build a more intuitive, visualized dashboard as shown below.\
For more details, please refer to [the official vLLM documentation](https://github.com/vllm-project/vllm/tree/main/examples/production_monitoring)!

![](asset/benchmark-screenshot.png)


## Cleaning Up

You can terminate a running Run using the `vessl run terminate` command.

```sh
# List runs to find the ID of the run to terminate
$ vessl run list

Organization: vessl-ai
Project: llm-demo-20240124
 ID            Name           Type         Status      Created                    Description
 ............  pdf-chatbot    batch        terminated  2024-01-25 01:37:52+00:00
 ............  pdf-chatbot    interactive  terminated  2024-01-25 01:47:11+00:00
 ............  vllm-demo      batch        terminated  2024-02-05 14:37:27+00:00
 ............  test-notebook  interactive  terminated  2024-02-05 14:47:10+00:00
 369367189168  vllm-demo      batch        running     2024-02-06 04:16:36+00:00

# Terminate the run
$ vessl run terminate 369367189168

Organization: vessl-ai
Project: llm-demo-20240124
Terminated '#369367189168'.
```

You can also terminate the Run by clicking Actions -> Terminate in the top right corner of the Web dashboard.

![](asset/run-terminate.png)

## Advanced Topics for Faster Deployment

The example describes a straightforward method to create a VESSL Run and check the endpoint using a single file. Here are some approaches you can apply to expedite the execution of Runs beyond the provided example.

### Build a Custom Docker Image for Faster Run Turn-Up

During the Run execution process, you can observe initialization tasks such as the installation of Prometheus and vLLM in the logs. Including these initialization tasks in the Docker image can speed up Run execution.

Below is an example of building a Docker image that includes various dependencies.

```Dockerfile
FROM quay.io/vessl-ai/torch:2.2.0-cuda12.3-r3

ENV PROMETHEUS_VERSION=2.49.1

WORKDIR /app

# Download and unzip prometheus binary to /app/prometheus
RUN wget https://github.com/prometheus/prometheus/releases/download/v$PROMETHEUS_VERSION/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz
RUN tar -xvf prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz
RUN mv prometheus-$PROMETHEUS_VERSION.linux-amd64 ./prometheus
RUN rm prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz

# Copy prometheus.yml
COPY monitoring/prometheus.yml /app/prometheus/prometheus.yml

# Install dependencies
COPY requirements.txt /

app/requirements.txt
RUN pip install -r /app/requirements.txt

# Entrypoint
ENTRYPOINT ["python", "-m", "api.py"]
```

### Caching `~/.cache/huggingface/hub` for Faster Model Loading

The example uses models from HuggingFace, which require downloading upon the first load, thus taking time. Pre-caching the `~/.cache/huggingface/hub` directory can reduce this loading time.

You can cache the `~/.cache/huggingface/hub` directory using VESSL Artifacts as follows. The directory is saved as Artifacts when the Run terminates, and can be reused in subsequent Runs.

![](asset/hf-cache-volumemount.png)

The YAML manifest will include additional `import` and `export` sections as shown below.

```yaml
import:
  ...
  /root/.cache/huggingface/hub/: vessl-artifact://{ORGANIZATION}/{PROJECT}/huggingface-models
export:
  /root/.cache/huggingface/hub/: vessl-artifact://{ORGANIZATION}/{PROJECT}/huggingface-models

```

This method can be applied not only to HuggingFace but also to caching necessary libraries, models, etc., using VESSL Artifact.
