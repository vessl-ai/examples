#Real-time Credit Scoring with Feast on AWS

## Summary
- This example is written on top of [feast-dev/feast-aws-credit-scoring-tutorial](https://github.com/feast-dev/feast-aws-credit-scoring-tutorial).
- Fixed minor errors and rewrote `feature_repo` to apply updated Feast features.
![credit-score-architecture@2x](https://user-images.githubusercontent.com/6728866/132927464-5c9e9e05-538c-48c5-bc16-94a6d9d7e57b.jpg)

## Requirements
- Terraform (v1.0 or later)
- AWS CLI (v2.2 or later)
  - Your aws credentials should be set in `~/.aws/credentials`.

## Setup
### Setting up AWS infra (Redshift and S3) with Terraform
We will deploy the following resources:
- Redshift cluster
- S3 bucket: zipcode and credit history parquet files
- IAM roles and policies: Redshift to access S3
- Glue catalogs: zipcode features and credit history

1. Initialize terraform
```bash
cd infra
terraform init
```
2. Set terraform variables
```bash
export TF_VAR_region="ap-northeast-2"
export TF_VAR_project_name="vessl-credit-scoring-project"
export TF_VAR_admin_password="MyAdminPassword1"
```
3. Plan and deploy your infrastructure
```bash
terraform plan
terraform apply
```
Once your infrastructure is deployed, you should see the following outputs from Terraform
```bash
redshift_cluster_identifier = "vessl-credit-scoring-project-redshift-cluster"
redshift_spectrum_arn = "arn:aws:iam::<Account>:role/s3_spectrum_role"
credit_history_table = "credit_history"
zipcode_features_table = "zipcode_features"
```
To have these outputs in env variables, you can source the `env` script
```bash
(cd .. && source env)
```
### Setting up Feast
Install Feast using pip
```bash
pip install feast[aws]
```
Deploy the feature store by running `apply` from within the `feature_repo/` directory.
```bash
cd feature_repo
feast apply
```
Once `feast apply` has finished, you can see the following created entities and deploying infrastructure statement.
```bash
Created entity dob_ssn
Created entity zipcode
Created feature view zipcode_features
Created feature view credit_history

Deploying infrastructure for zipcode_features
Deploying infrastructure for credit_history
```
Next we load features into the online store using materialize command.
```bash
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
feast materialize "2013-01-01" $CURRENT_TIME
```
```bash
Materializing 2 feature views from 2013-01-01 09:00:00+09:00 to 2023-01-30 09:32:29+09:00 into the dynamodb online store.

zipcode_features:
01/30/2023 09:34:12 AM botocore.credentials INFO: Found credentials in shared credentials file: ~/.aws/credentials
100%|███████████████████████████████████████████████████████| 28844/28844 [00:25<00:00, 1134.96it/s]
credit_history:
100%|███████████████████████████████████████████████████████| 28633/28633 [00:27<00:00, 1043.41it/s]
```
Return to the root of the repository
```bash
cd ..
```

## Train and test the model
Finally, we train the model using a combination of loan data from S3 and our zipcode and credit history features from
Redshift (which in turn queries S3), and then we test online inference by reading those same features from DynamoDB.
```bash
python run.py
```
The script should then output the result of a single loan application
```bash
loan rejected!
```

## Interactive demo (using Streamlit)
Once the credit scoring model has been trained it can be used for interactive loa application using Streamlit.
Simply start the Streamlit application.
```bash
streamlit run app.py
```
Then navigate to the URL on which Streamlit is being served. You should see a user interface through which 
loan applications can be made:
![Streamlit screenshot](asset/streamlit.png)

## Destroy the deployed infrastructure
```bash
cd infra
terraform destroy 
```
