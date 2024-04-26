# 1-minute guide to VESSL Serve

## What's included
* `api.py` - Creates a FastAPI-enabled text generation app that communicates with vLLM-accelerated Mistral-7B.
* `quickstart.yaml` - Defines the service spec for the API server such as compute options, autoscaling, port routing, and more
* `requirements.txt` - Lists the additional dependencies that are not included in the pre-built Docker image

## Launch the app
You can launch the service using the `vessl serve` command. This rolls out a new "revision" of your model in production.
```
vessl serve revision create -f quickstart.yaml
```
In the background, VESSL Serves offloads the common logic required for deploying models on the cloud.
* Spin up a GPU-accelerated workload and set up a service environment
* Push the model and the API scripts to the cloud
* Create an API server with a dedicated port for receiving inference requests

## Using the app
Once the instance gets up and running, you can interact with the model using the API endpoint. 

he app generates responses based on the input JSON request which accepts input to the language model as `"prompt"`. This is defined under `/generate` in `api.py`. Try out the following `curl` command to see the app in action. Make sure to change the endpoint URL with your own. 
```
curl -X 'POST' \
  'https://model-service-gateway-1l4g3zwut6xt.oregon.google-cluster.vessl.ai/generate' \
  -H 'accept: application/json' \
  -d '{"prompt": "Can you explain the background concept of LLM?"}'
```

Click the endpoint URL to see how you can interact with the app, using additional input JSON fields, for example. 
![](assets/fastapi.png)

VESSL Serve offloads the complex challenges of deploying custom models while ensuring availability, scalability, and reliability.
* Autoscale the model to handle peak loads and scale to zero when it's not being used
* Route traffic to different versions of the model
* Monitor predictions and key metrics in real-time using dashboards & logs
