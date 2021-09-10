# MNIST
Run MNIST example on SavviHub:
## PyTorch
* Dataset mount
  1. Create a new dataset with a public S3 bucket directory `s3://savvihub-public-apne2/mnist`.
  2. Mount the dataset to `/input` at the experiment create form.
* Start Command
  ```bash
  pip install -r mnist/pytorch/requirements.txt && python mnist/pytorch/main.py --save-model --save-image
  ```
* Environment variables
  ```bash
  epochs
  optimizer
  batch_size
  learnig_rate
  ```
## Keras
* No dataset needed.
* Start Command
  ```bash
  pip install -r mnist/keras/requirements.txt && python mnist/keras/main.py --save-model --save-image
  ```
* Environment variables
  ```bash
  epochs
  optimizer
  batch_size
  learning_rate
  ```