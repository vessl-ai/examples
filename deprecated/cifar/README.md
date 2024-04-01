# CIFAR
Run CIFAR example on [VESSL](https://vessl.ai):
> Noted that you should add [hyperparameters](../README.md) as arguments to the start command

## PyTorch
### Start Command
  ```bash
  pip install -r examples/cifar/pytorch/requirements.txt && python examples/cifar/pytorch/main.py --save-model
  ```
### Hyperparameters
  ```bash
  l1 # [default: 2]
  l2 # [default: 2]
  lr # [default: 0.01]
  batch_size # [default: 128]
  epochs # [default: 10]
  ```