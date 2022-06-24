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