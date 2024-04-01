# MNIST

Run MNIST example on [VESSL](https://vessl.ai).

This example will walk through how to
- train model on VESSL Run
- save model in VESSL Models
- serve model using VESSL Serve.

## Train Model Locally

Train model with the following command.
```sh
# PyTorch
pip install -r pytorch/requirements.txt
python examples/mnist/pytorch/main.py --save-model --save-image 

# Keras
pip install -r examples/mnist/keras/requirements.txt
python examples/mnist/keras/main.py --save-model --save-image 
```

### Hyperparameters
```sh
epochs # [defaults: 10]
optimizer # adam, sgd, adadelta [defaults: adam]
batch_size # [defaults: 128]
learning_rate # [defaults: 0.01]
```


## Train Model on VESSL

Use VESSL Run to train model on VESSL's clusters (or any other cluster you configure on VESSL).
```sh
# PyTorch
vessl run "pip install -r pytorch/requirements.txt; python pytorch/main.py --save-model --save-image"

# Keras
vessl run "pip install -r keras/requirements.txt; python keras/main.py --save-model --save-image"
```

## Saving Model to VESSL model registry (PyTorch)

### Save the Model
1. Create a model repository in [VESSL model registry](https://docs.vessl.ai/user-guide/model-registry/creating-a-model) if you don't have one.
    - Use model repository name as `{VESSL_MODEL_REPO_NAME}` in the following steps.
2. Train a PyTorch model with `--save-model` option.
    - Use `{SAVED_MODEL_PATH}` as the path to the saved model in the following steps.
3. Run a script to register the model:

```sh
# PyTorch
python pytorch/model.py --checkpoint {SAVED_MODEL_PATH} --model-repository {VESSL_MODEL_REPO_NAME}
```

This will register the model to the model repository and yield the following output:
```sh
Successfully registered model: https://vessl.ai/{ORGANIZATION_NAME}/models/{VESSL_MODEL_REPO_NAME}/{NUMBER}
```
Keep the number in the URL as `{MODEL_NUMBER}` in the following steps.

### Serve Model in Local

Run the following command to serve the model previously registered.
```sh
vessl model serve vessl-mnist-example {MODEL_NUMBER} --install-reqs
```

After deploying the registered model, you can curl HTTP POST request with sample mnist image.
```sh
curl -X POST --data-binary @pytorch/sample_img/mnist_7.png localhost:8000
```

This will yield the following output:
```sh
{"result": 7}
```

### Serve Model in remote using VESSL Serving
See [VESSL Serving](https://docs.vessl.ai/user-guide/serving) docs for more details.
