#Real-time Credit Scoring with Feast on AWS

## Summary
- This example is written on top of [feast-dev/feast-aws-credit-scoring-tutorial](https://github.com/feast-dev/feast-aws-credit-scoring-tutorial).
- Fixed minor errors and rewrote `feature_repo` to apply updated Feast features.
![credit-score-architecture@2x](https://user-images.githubusercontent.com/6728866/132927464-5c9e9e05-538c-48c5-bc16-94a6d9d7e57b.jpg)

## Requirements
- Terraform (v1.0 or later)
- AWS CLI (v2.2 or later)

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