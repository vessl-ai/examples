# Distributed CIFAR
Run Distributed CIFAR example on Vessl:
* Dataset mount
  1. Create a new dataset with a public S3 bucket directory `https://savvihub-public-apne2.s3.ap-northeast-2.amazonaws.com/cifar-10`.
  2. Mount the dataset to `/input` at the experiment create form.

## PyTorch
* Start Command
  ```bash
  python -m torch.distributed.launch  \
    --nproc_per_node=1  \
    --nnodes=2  \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    examples/distributed_cifar/pytorch/main.py
  ```
  * User-defined options
    * `--nproc_per_node`: the number of GPUs per node
    * `--nnodes`: the number of nodes
  * Vessl-defined options
    * `--node_rank`: vessl will automatically set `$RANK` as an environment variable
    * `--master_addr`: vessl will automatically set `$MASTER_ADDR` as an environment variable
    * `--master_port`: vessl will automatically set `$MASTER_PORT` as an environment variable
