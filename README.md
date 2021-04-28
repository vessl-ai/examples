# examples
This repository contains SavviHub examples. If you want to learn more about SavviHub please follow the [quick start documentation](https://docs.savvihub.com/quick-start).

## MNIST
Run MNIST example and save model:
1. PyTorch
* Dataset mount: `s3://savvihub-public-apne2/mnist` -> `/input`
```bash
pip install -r mnist/requirements.txt && python mnist/pytorch/main.py --save-model
```
2. Keras
* No dataset needed.
```bash
pip install -r mnist/requirements.txt && python mnist/keras/main.py --save-model
```

## Detectron2

Run Detectron2 example:
* Dataset mount: `s3://savvihub-public-apne2/detectron2` -> `/input`
```bash
pip install -r detectron2/requirements.txt && python detectron2/main.py
```