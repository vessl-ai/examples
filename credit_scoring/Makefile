run: install-feast feast-apply install-requirements train-model

install-feast:
	@echo "Installing Feast..."
	pip install 'feast[aws]'

feast-apply:
	@echo "Feast apply..."
	tf_redshift_cluster_identifier=vessl-credit-scoring-project-redshift-cluster tf_redshift_spectrum_arn=arn:aws:iam::962345818314:role/s3_spectrum_role feast -c feature_repo apply

install-requirements:
	@echo "Installing requirements..."
	pip install -r requirements.txt

train-model:
	@echo "Start training..."
	tf_redshift_cluster_identifier=vessl-credit-scoring-project-redshift-cluster tf_redshift_spectrum_arn=arn:aws:iam::962345818314:role/s3_spectrum_role python main.py
