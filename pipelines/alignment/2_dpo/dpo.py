import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


def main(
    script_args: ScriptArguments, training_args: DPOConfig, model_config: ModelConfig
):
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
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )

    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            trust_remote_code=model_config.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_model = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    try:
        dataset = load_from_disk(script_args.dataset_name)
    except FileNotFoundError:
        dataset = load_dataset(script_args.dataset_name)

    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = (
        dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None
    )

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()

    main(script_args, training_args, model_config)