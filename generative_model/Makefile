run: install-diffuser install-requirements fine-tuning-model

install-diffuser:
	@echo "Install diffuser dev version..."
	cd diffusers; pip install .

install-requirements:
	@echo "Installing requirements..."
	pip install -r examples/generative_model/requirements.txt

fine-tuning-model:
	@echo "Fine-tuning diffuser by LoRa on VESSL"
	accelerate launch --mixed_precision="bf16" --multi_gpu \
	examples/generative_model/LoRa.py --random_flip --train_batch_size=${train_batch_size} \
	--num_train_epochs=$(num_train_epochs) --learning_rate=${learning_rate} --seed=${seed} \
	--output_dir="/output" --validation_prompt=${validation_prompt} --on_vessl=True