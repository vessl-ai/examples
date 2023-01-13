# Recommenders
Run recommenders example on [VESSL](https://vessl.ai):

## Keras
### Dataset mount
* MovieLens (https://grouplens.org/datasets/movielens/latest/) dataset is required.
* You can use VESSL's public S3 dataset. `s3://vessl-public-apne2/movie-lens`
* Mount the dataset to `/input` at the experiment create form.
### Start Command
  ```bash
  pip install -r examples/recommenders/keras/requirements.txt && python examples/recommenders/keras/main.py --save-model
  ```


## SasRec & SSE-PT

### Environments
* Cuda >= 11.2
* Python >= 3.7

### Dataset mount 
* Amazon beauty(http://jmcauley.ucsd.edu/data/amazon/index.html) dataset  
* You can use VESSL's public S3 dataset `s3://vessl-public-apne2/amazon-ranking/`
* Mount the dataset to `/input` at the experiment create form.

### Start Command 
  ```bash
  pip install -r examples/recommenders/sasrec/requirements.txt && python examples/recommenders/sasrec/main.py --lr $lr 
  --batch-size $batch-size --num-epochs $num_epochs 
  ```