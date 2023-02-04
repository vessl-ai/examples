define ARC_ENV
TF_VAR_region="ap-northeast-2"
TF_VAR_project_name="vessl-credit-scoring-project"
TF_VAR_admin_password="MyAdminPassword1"
tf_redshift_spectrum_arn=arn:aws:iam::962345818314:role/s3_spectrum_role
tf_redshift_cluster_identifier=vessl-credit-scoring-project-redshift-cluster
tf_credit_history_table=credit_history
tf_zipcode_features_table=zipcode_features
endef
export ARC_ENV

run: install-feast feast-apply install-requirements train-model

install-feast:
	@echo "Installing Feast..."
	pip install 'feast[aws]'

feast-apply:
	@echo "Feast apply..."
	feast -c examples/credit_scoring/feature_repo apply

install-requirements:
	@echo "Installing requirements..."
	pip install -r examples/credit_scoring/requirements.txt

train-model:
	@echo "Start training..."
	python examples/credit_scoring/main.py
