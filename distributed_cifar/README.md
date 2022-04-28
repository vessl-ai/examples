# Distributed CIFAR
Run Distributed CIFAR example on [VESSL](https://vessl.ai):
> Noted that you should add [hyperparameters](../README.md) as arguments to the start command

## PyTorch
### Dataset mount
  1. Create a new dataset with a public S3 bucket directory `s3://vessl-public-apne2/cifar-10`.
  2. Mount the dataset to `/input` at the experiment create form.
### Start Command
  ```bash
  python -m torch.distributed.launch \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$NUM_TRAINERS \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    examples/distributed_cifar/pytorch/main.py
  ```
### VESSL-defined options
  VESSL will automatically set the following environment variables. 
  * `$NUM_NODES`
  * `$NUM_TRAINERS`
  * `$RANK`
  * `$MASTER_ADDR` 
  * `$MASTER_PORT` 
