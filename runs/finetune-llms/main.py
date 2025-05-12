from transformers import HfArgumentParser
from trl import SFTConfig, SFTTrainer
from vessl.integration.transformers import VesslCallback

from arguments import DatasetArguments, ModelArguments, PeftArguments, VesslArguments
from dataset import create_dataset
from model import get_peft_config, get_unsloth_peft_model, load_model_and_tokenizer


def main(
    model_args: ModelArguments,
    peft_args: PeftArguments,
    data_args: DatasetArguments,
    training_args: SFTConfig,
    vessl_args: VesslArguments,
):
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    train_dataset, eval_dataset = create_dataset(data_args)

    peft_config = None
    if peft_args.peft_type is not None:
        if model_args.use_unsloth:
            model = get_unsloth_peft_model(model, peft_args, training_args)
        else:
            peft_config = get_peft_config(peft_args)

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=[
            VesslCallback(
                upload_model=vessl_args.upload_model,
                repository_name=vessl_args.repository_name,
            )
        ],
    )

    model.config.use_cache = False

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    if vessl_args.save_merged:
        model.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit")

if __name__ == "__main__":
    parser = HfArgumentParser(
        (
            ModelArguments,
            PeftArguments,
            DatasetArguments,
            SFTConfig,
            VesslArguments,
        )
    )
    model_args, peft_args, data_args, training_args, vessl_args = (
        parser.parse_args_into_dataclasses()
    )

    main(model_args, peft_args, data_args, training_args, vessl_args)
