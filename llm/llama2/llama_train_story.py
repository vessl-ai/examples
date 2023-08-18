import argparse

# Custom Kwargs handler for increasing timeout
from datetime import timedelta

import torch
import vessl
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)


# Custom Dataset for VESSL Docs
class TextDataset(Dataset):
    """
    Dataset wrapper for pure text.
    """

    def __init__(self):
        self.data = load_from_disk("/data/tolkien_256")
        self.processed = []
        for e in self.data["train"]["input_ids"]:
            inst = {"input_ids": e, "labels": e}
            self.processed.append(inst)

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        return self.processed[idx]


def collate_fn(samples):
    inputs = torch.stack([torch.tensor(sample["input_ids"]) for sample in samples])
    labels = torch.stack([torch.tensor(sample["labels"]) for sample in samples])
    atts = torch.stack([torch.ones(256) for sample in samples])
    return {"input_ids": inputs, "labels": labels, "attention_mask": atts}


def train():
    parser = argparse.ArgumentParser()
    # set hyperparameters
    parser.add_argument("--lora_r", type=float, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=1)
    args = parser.parse_args()

    train_dataset = TextDataset()
    tokenizer = LlamaTokenizer.from_pretrained("/data/llama-2-7b-hf/", legacy=False)
    tokenizer.pad_token = tokenizer.unk_token
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn
    )

    # for longer timeouts
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=6000))
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    # vessl configuration
    if accelerator.is_main_process:
        vessl.init()
    accelerator.wait_for_everyone()

    # peft configuration
    print(f"lora_r: {args.lora_r}")
    print(f"lora_alpha: {args.lora_alpha}")
    print(f"lora_dropout: {args.lora_dropout}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    seed = 42
    set_seed(seed)

    model = AutoModelForCausalLM.from_pretrained("/data/llama-2-7b-hf/")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_loader) * args.num_epoch),
    )
    model, train_loader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_loader, optimizer, lr_scheduler
    )

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    for epoch in range(args.num_epoch):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader)):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_epoch_loss = total_loss / len(train_loader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        # vessl logging
        if accelerator.is_main_process:
            train_epoch_loss = total_loss / len(train_loader)
            vessl.log(
                step=epoch,
                payload={
                    "loss": train_epoch_loss,
                    "ppl": train_ppl,
                    "learning_rate": lr,
                },
            )
        accelerator.wait_for_everyone()

        # evaluation
        if epoch % args.save_interval == 0 or epoch == args.num_epoch - 1:
            model.eval()
            with torch.no_grad():
                unwrapped_model = accelerator.unwrap_model(model)
                logging_str = f"{args.lora_r}_{args.lora_alpha}_{args.lora_dropout}_{args.lr}_{epoch}"
                accelerator.wait_for_everyone()
                unwrapped_model.save_pretrained(
                    f"/output/{logging_str}",
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
                )

                accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
