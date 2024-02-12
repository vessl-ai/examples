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


## SasRec

### Environments
* Cuda >= 11.2
* Python >= 3.8

### Dataset mount 
* Amazon beauty(http://jmcauley.ucsd.edu/data/amazon/index.html) dataset  
* You can use VESSL's public HuggingFace dataset `https://huggingface.co/datasets/VESSL/amazon-beauty-dataset`
* Mount the dataset to `/input` at the experiment create form. 

### Start Command 
#### Training
  ```bash
  pip install -r examples/recommenders/sasrec/requirements.txt && python examples/recommenders/sasrec/main.py
  ```
#### Registering model(default) for serving
  ```bash
  pip install -r examples/recommenders/sasrec/requirements.txt && python examples/recommenders/sasrec/model.py
  ```

### Serving in VESSL
* Example input data
  * Create a csv file (e.g. `input_data.csv`) as follows and curl to the endpoint with authentication token that you can find on the serving page. 
    ```bash
    [1, 12, 123, 13, 5]
    ```
    ```bash
    curl -X POST -H "X-AUTH-KEY:[YOUR-AUTHENTICATION-TOKEN]" -d @input_data.csv https://service-XXXX.apne2-prod1-cluster.savvihub.com
    ```
  * Example output
    ```bash 
    +--------+----------+--------------------+
    | Rank   | ItemID   | Similarity Score   |
    |--------+----------+--------------------|
    | 1      | 195      | 1.5078533          |
    | 2      | 686      | 1.4597058          |
    | 3      | 929      | 1.3184035          |
    | 4      | 80       | 1.2633942          |
    | 5      | 124      | 1.2385566          |
    +--------+----------+--------------------+
    
    Result: item 195
    ```
    
### Serving with Streamlit
#### Requirements
* Streamlit
* Trained model
  * best.data-00000-of-00001
  * best.index
#### HuggingFace Space
You can run an inference demo in [VESSL HuggingFace Space](https://huggingface.co/spaces/VESSL/recommender) with sample model.
