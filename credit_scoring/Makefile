run: install-requirements train-model

install-requirements:
	@echo "Installing requirements..."
	pip install -r requirements.txt

train-model:
	@echo "Start training..."
	python main.py
	cp -r /feature_repo /output && cp -r ~/.aws /output

create-model:
	@echo "Create model..."
	python model.py
