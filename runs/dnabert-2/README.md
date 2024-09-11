# DNABERT-2

This repository contains code for [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2) model finetuning for COVID-19 variant classification, and a sample Gradio web app to use the finetuned model for inference.

DNABERT-2 is a refined genome foundation model that adapts an efficient tokenizer and employs multiple strategies to overcome input length constraints, reduce time and memory expenditure, and enhance model capability. It achieves comparable performance to the state-of-the-art model with 21× fewer parameters and approximately 92× less GPU time in pre-training.

## Finetuning
### Running Locally

1. Clone the repository:
    ```bash
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/dnabert-2
    ```

2. Install required dependencies:
    ```bash
    $ pip install -r requirements-train.txt
    ```

3. Download [GUE benchmark dataset](https://drive.google.com/file/d/1GRtbzTe3UXYF1oW27ASNhYX3SZ16D7N2/view?usp=sharing) and unzip it.

4.  Run the script. Change `${your_path_to_data}` to the path where you unzipped the data.
    > Current script is set to use `DataParallel` for training on 4 GPUs. If you have different number of GPUs, please change the `per_device_train_batch_size` and `gradient_accumulation_steps` accordingly to adjust the global batch size to 32.
    ```bash
    $ lr=3e5
    $ python train.py \
        --model_name_or_path zhihan1996/DNABERT-2-117M \
        --data_path ${your_path_to_data}/GUE/virus/covid \
        --kmer -1 \
        --run_name DNABERT2_${lr}_virus_covid \
        --model_max_length 256 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 8 \
        --fp16 \
        --save_steps 200 \
        --output_dir output \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --logging_steps 100000 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False
    ```

### Running with VESSL Run
VESSL is a platform for deploying and managing AI applications. It allows you to deploy your AI applications on the cloud with a single command, and provides a web interface for managing your applications.

To start finetuning with VESSL Run, follow the steps below:
1. Create a new project on VESSL.
2. Create a new model repository on VESSL.
3. Clone the repository:
    ```bash
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/dnabert-2
    ```
4. Update the `export` section of the provided YAML file ([`run-covid-finetuning.yaml`](./run-covid-finetuning.yaml)) with the model repository path you created above:
    ```yaml
    # run-covid-finetuning.yaml
    ...
    export:
      /output/: vessl-model://{organization}/{model repository name}
    ...
    ```
5. Create a VESSL Run with the YAML file you updated:
    ```bash
    $ vessl run create -f run-covid-finetuning.yaml
    ```
6. You can see the full log and the metrics plot at the Run info page.

For additional information and support, please refer to the [VESSL documentation](https://docs.vessl.ai).

## Inference
### Running locally
1. Clone the repository if you haven't yet:
    ```bash
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/dnabert-2
    ```
2. Install required dependencies:
    ```bash
    $ pip install -r requirements-inference.txt
    ```
3. Update the environment variable `MODEL_PATH` with the path of the model:
    ```bash
    $ export MODEL_PATH={your_model_path}
    ```
4. Run the sample gradio web app:
    ```bash
    $ python inference.py
    ```
5. Access the Gradio interface by opening http://localhost:7860 in your web browser.

### Deploying with VESSL Run
1. Create a new project on VESSL if you haven't yet.
2. Clone the repository if you haven't yet:
    ```bash
    $ git clone https://github.com/vessl-ai/examples.git
    $ cd examples/dnabert-2
    ```
3. Update the `import` section of the provided YAML file ([`run-inference.yaml`](./run-inference.yaml)) with the model you created above:
    ```yaml
    # run-inference.yaml
    ...
    import:
    ...
      /model/: vessl-model://{organization}/{model repository name}/{model number}
    ...
    ```
4. Update the `MODEL_PATH` environment variable section of the YAML file if you need to:
    ```yaml
    # run-inference.yaml
    ...
    env:
      MODEL_PATH: /model/checkpoint-4400  # you might want to change the step number based on your best model
    ```
5. Deploy the application with the YAML file:
    ```bash
    $ vessl run create -f run-inference.yaml
    ```
6. Access the Gradio interface via link on the endpoint section of the Run summary page.

For additional information and support, please refer to the [VESSL documentation](https://docs.vessl.ai).

## Citation
```bibtex
@misc{zhou2023dnabert2,
      title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome}, 
      author={Zhihan Zhou and Yanrong Ji and Weijian Li and Pratik Dutta and Ramana Davuluri and Han Liu},
      year={2023},
      eprint={2306.15006},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
```