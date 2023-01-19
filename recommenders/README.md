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
* You can use VESSL's public S3 dataset `s3://vessl-public-apne2/amazon-ranking/`
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

### Serving 
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
    Recommended item numbers and their similarity scores(not normalized)
    195 : 1.5078533
    686 : 1.4597058 
    929 : 1.3184035
    80 : 1.2633942 
    124 : 1.2385566
    129 : 1.2012779
    2155 : 1.1481848
    596 : 1.1317025
    746 : 1.1095992
    2634 : 1.0377761
    Result: item 195
    ```