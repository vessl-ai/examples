run: install-feast feast-apply install-requirements train-model

install-feast:
	@echo "Installing Feast..."
	pip install 'feast[aws]'

feast-apply:
	@echo "Feast apply..."
	feast -c feature_repo apply

install-requirements:
	@echo "Installing requirements..."
	pip install -r requirements.txt

train-model:
	@echo "Start training..."
	python main.py
