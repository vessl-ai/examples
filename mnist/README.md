# MNIST
Run MNIST example on Vessl:
## PyTorch
* Dataset mount
  1. Create a new dataset with a public S3 bucket directory `s3://savvihub-public-apne2/mnist`.
  2. Mount the dataset to `/input` at the experiment create form.
* Start Command
  ```bash
  pip install -r examples/mnist/pytorch/requirements.txt && python examples/mnist/pytorch/main.py --save-model --save-image
  ```
* Environment variables
  ```bash
  epochs
  optimizer
  batch_size
  learning_rate
  ```
## Keras
* No dataset needed.
* Start Command
  ```bash
  pip install -r examples/mnist/keras/requirements.txt && python examples/mnist/keras/main.py --save-model --save-image
  ```
* Environment variables
  ```bash
  epochs
  optimizer
  batch_size
  learning_rate
  ```