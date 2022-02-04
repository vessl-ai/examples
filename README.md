# VESSL examples
This repository contains [VESSL](https://www.vessl.ai/) examples. If you want to learn more about VESSL please follow the [quick start documentation](https://docs.savvihub.com/quick-start).

## Use hyperparameters on VESSL
[Hyperparameters](https://docs.vessl.ai/user-guide/experiment/creating-an-experiment#hyperparameters) are automatically add to the container as [environment variables](https://kubernetes.io/docs/tasks/inject-data-application/define-environment-variable-container/) with the given key and value. If you want to use them at runtime, then append them to the [start command](https://docs.vessl.ai/user-guide/experiment/creating-an-experiment#start-command) as follows.
```bash
# Add learning_rate as a hyperparameter
python main.py --learning-rate $learning_rate 
```

You can now take the desired type of argument in Python script at runtime.
```Python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', type=float, default=0.01)
args = parser.parse_args()
```

## Try out VESSL examples

- [Image classification (CIFAR) using Convnets (PyTorch)](cifar)
- [Object detection with balloon dataset using Detectron2 (PyTorch)](detectron2)
- [Distributed training for image classification (CIFAR) with Resnet18](distributed_cifar)
- [Language modeling using LSTM RNNs (PyTorch)](language_model)
- [Image classification (MNIST) using Convnets (PyTorch, Keras)](mnist)
