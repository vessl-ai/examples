from dataclasses import dataclass

import torch
from accelerate import PartialState
from datasets import Value, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

FORMAT_MAPPINGS = {
    "chatml": [
        {
            "content": Value(dtype="string", id=None),
            "role": Value(dtype="string", id=None),
        }
    ],
    "prompt_only": {
        "prompt": Value(dtype="string", id=None),
    },
}


@dataclass
class AdditionalArguments:
    max_length: int = 512


def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize_chatml(element):
        input_ids = tokenizer.apply_chat_template(
            element["messages"][:1],
            padding=False,
            add_generation_prompt=True,
        )
        return {"input_ids": input_ids, "lengths": len(input_ids)}

    def tokenize_prompt_only(element):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": element["prompt"]}],
            padding=False,
            add_generation_prompt=True,
        )
        return {"input_ids": input_ids, "lengths": len(input_ids)}

    if dataset.features.get("messages") == FORMAT_MAPPINGS["chatml"]:
        return dataset.map(
            tokenize_chatml,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )
    elif dataset.features.items() >= FORMAT_MAPPINGS["prompt_only"].items():
        return dataset.map(
            tokenize_prompt_only,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )
    else:
        raise ValueError(f"Unsupported dataset format: {dataset.features}")


def main(
    script_args: ScriptArguments,
    training_args: PPOConfig,
    model_config: ModelConfig,
    additional_args: AdditionalArguments,
):
    # get moddel configuration
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ("auto", None)
        else getattr(torch, model_config.torch_dtype)
    )

    quant_config = get_quantization_config(model_config)

    model_kwargs = {
        "attn_implementation": model_config.attn_implementation,
        "torch_dtype": torch_dtype,
        "use_cache": False if training_args.gradient_checkpointing else True,
        "device_map": get_kbit_device_map() if quant_config is not None else None,
        "quantization_config": quant_config,
    }

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # models
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_config.trust_remote_code,
        num_labels=1,
        **model_kwargs,
    )
    value_model = reward_model

    peft_config = get_peft_config(model_config)

    # dataset
    try:
        dataset = load_from_disk(script_args.dataset_name)
    except FileNotFoundError:
        dataset = load_dataset(script_args.dataset_name)

    if script_args.dataset_test_split not in dataset.keys():
        dataset = dataset["train"].train_test_split(test_size=0.1)
        dataset[script_args.dataset_test_split] = dataset.pop("test")

    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split]

    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        if eval_dataset is not None:
            eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        # filtering
        train_dataset = train_dataset.filter(
            lambda x: x["lengths"] <= additional_args.max_length,
            num_proc=training_args.dataset_num_proc,
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(
                lambda x: x["lengths"] <= additional_args.max_length,
                num_proc=training_args.dataset_num_proc,
            )

    # trainer
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, PPOConfig, ModelConfig, AdditionalArguments))
    script_args, training_args, model_config, additional_args = (
        parser.parse_args_into_dataclasses()
    )
    main(script_args, training_args, model_config, additional_args)
