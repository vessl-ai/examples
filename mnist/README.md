# MNIST
Run MNIST example on [VESSL](https://vessl.ai):
> Noted that you should add [hyperparameters](../README.md) as arguments to the start command

## PyTorch
### Dataset mount
  1. Create a new dataset with a public S3 bucket directory `s3://vessl-public-apne2/mnist`.
  2. Mount the dataset to `/input` at the experiment create form.
### Start Command
  ```bash
  pip install -r examples/mnist/pytorch/requirements.txt && python examples/mnist/pytorch/main.py --save-model --save-image 
  ```
### Hyperparameters
```bash
epochs # [defaults: 5]
optimizer # adadelta, sgd [defaults: adadelta]
batch_size # [defaults: 128]
learning_rate # [defaults: 0.1]
```
### Serving
#### 1. Create a model
Once you finished training, [create a model](https://docs.vessl.ai/user-guide/model-registry/creating-a-model) with the `model.pt` file in the VESSL Web Console.
#### 2. Register a model 
On your local machine, change directory to the `vessl-ai/examples/mnist/pytorch`. Then, replace the `repository_name` and `model_number` parameter of `vessl.register()` method with your model repository name and model number in `model.py`. You can [register the model](https://docs.vessl.ai/user-guide/model-registry/deploying-a-model) by running `model.py`.   
```bash
cd examples/mnist/pytorch
python model.py
```
#### 3. Deploy a model
After deploying the registered model on Web Console, you can curl HTTP POST request with sample mnist image.
```bash
curl -X POST -H "X-AUTH-KEY:[YOUR-AUTHENTICATION-TOKEN]" --data-binary @sample/mnist_7.png [SERVICE ENDPOINT]
```


## Keras
### Dataset mount  
* No dataset needed
### Start Command
  ```bash
  pip install -r examples/mnist/keras/requirements.txt && python examples/mnist/keras/main.py --save-model --save-image 
  ```
### Hyperparameters
```bash
epochs # [defaults: 10]
optimizer # adam, sgd, adadelta [defaults: adam]
batch_size # [defaults: 128]
learning_rate # [defaults: 0.01]
```