# Fraud detection
Run fraud detection example on VESSL:
## Keras
* Dataset mount
    1. Create a new dataset with a public S3 bucket directory `s3://vessl-public-apne2/creditcard-undersample/`.
    2. Mount the dataset to `/input` at the experiment create form.
* Start Command
  ```bash
  python examples/fraud_detection/keras/main.py --save-model --save-image
  ```
* Environment variables
  ```bash
  epochs
  ```
