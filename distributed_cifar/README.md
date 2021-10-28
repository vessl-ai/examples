# Distributed CIFAR
Run Distributed CIFAR example on Vessl:
* Dataset mount
  1. Create a new dataset with a public S3 bucket directory `s3://savvihub-public-apne2/cifar-10`.
  2. Mount the dataset to `/input` at the experiment create form.

## PyTorch
* Start Command
  ```bash
  python -m torch.distributed.launch \
    --nnodes=$NUM_WORKERS \
    --nproc_per_node=$NUM_GPUS_PER_WORKER \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    examples/distributed_cifar/pytorch/main.py
  ```
  * Vessl-defined options
    * `--nnodes`: vessl will automatically set `$NUM_WORKERS` as an environment variable
    * `--nproc_per_node`: vessl will automatically set `$NUM_GPUS_PER_WORKER` as an environment variable
    * `--node_rank`: vessl will automatically set `$RANK` as an environment variable
    * `--master_addr`: vessl will automatically set `$MASTER_ADDR` as an environment variable
    * `--master_port`: vessl will automatically set `$MASTER_PORT` as an environment variable
