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
