run: install-requirements train-model

install-requirements:
	@echo "Installing requirements..."
	pip install -r requirements.txt

train-model:
	@echo "Start training..."
	python main.py
	cp -r /feature_repo /output

register-model:
	@echo "Register your model..."
	python model.py
