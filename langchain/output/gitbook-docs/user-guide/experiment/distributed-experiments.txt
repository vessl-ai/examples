---
description: Early access feature
---

# Distributed Experiments

{% hint style="info" %}
Only the PyTorch framework is supported distributed experiment currently.
{% endhint %}

### What is a distributed experiment?

A **distributed experiment** is a single machine learning run on top of multi-node or multi-GPUs. The distributed experiment results are consist of logs, metrics, and artifacts for each worker which you can find under corresponding tabs.

{% hint style="danger" %}
**Caveats**

Multi-node training is not always an optimal solution. We recommend you try several experiments with a few epochs to see if multi-node training is the correct choice for you.
{% endhint %}

#### Environment variables

VESSL automatically sets the below environment variables based on the configuration.

`NUM_NODES`: Number of workers

`NUM_TRAINERS`: Number of GPUs per node

`RANK`: The global rank of node

`MASTER_ADDR`: The address of the master node service

`MASTER_PORT`: The port number on the master address

### Creating a distributed experiment

#### Using Web Console

Running a distributed experiment on the web console is similar to a single node experiment. To create a distributed experiment, you only need to specify the number of workers. Other options are the same as those of a single node experiment.

{% content-ref url="creating-an-experiment.md" %}
[creating-an-experiment.md](creating-an-experiment.md)
{% endcontent-ref %}

#### Using CLI

To run a distributed experiment using CLI, the number of nodes must be set to an integer greater than one.

```bash
vessl experiment create --worker-count 2 --framework-type pytorch
```

### Examples: Distributed CIFAR

You can find the full example codes [here](https://github.com/savvihub/examples/tree/main/distributed\_cifar).

#### Step 1: Prepare CIFAR-10 dataset

Download the CIFAR dataset with the scripts below. and add a vessl type dataset to your organization.

```bash
wget -c --quiet https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
```

{% content-ref url="../dataset/adding-new-datasets.md" %}
[adding-new-datasets.md](../dataset/adding-new-datasets.md)
{% endcontent-ref %}

Or, you can simply add an AWS S3 type dataset to your organization with the following public bucket URI.

```
s3://savvihub-public-apne2/cifar-10
```

#### Step 2: Create a distributed experiment

To run a distributed experiment we recommend to use [`torch.distributed.launch`](https://pytorch.org/docs/stable/distributed.html) package. The example start command that runs on two nodes and one GPU for each node is as follows.

```
python -m torch.distributed.launch  \
  --nnodes=$NUM_NODES  \
  --nproc_per_node=$NUM_TRAINERS  \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  examples/distributed_cifar/pytorch/main.py
```

VESSL will automatically set environment variables of `--node_rank`, `--master_addr`, `--master_port`, `--nproc_per_node` and `--nnodes`.

### Files

In a distributed experiment, all workers share an output storage. Please be aware that files can be overrided by other workers when you use same output path.
