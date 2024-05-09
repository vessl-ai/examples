from datasets import load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

from arguments import DatasetArguments


def alpaca_to_chatml(sample):
    return {
        "messages": [
            {"role": "system", "content": sample["input"]},
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["output"]},
        ]
    }


def create_dataset(data_args: DatasetArguments):
    try:
        dataset = load_dataset(data_args.dataset_name)
    except DatasetGenerationError:
        dataset = load_from_disk(data_args.dataset_name)

    if set(dataset.column_names["train"]) >= {"input", "instruction", "output"}:
        dataset = dataset.map(alpaca_to_chatml)

    train_dataset = dataset["train"]
    if "val" in dataset.keys():
        eval_dataset = dataset["val"]
    else:
        eval_dataset = None

    return train_dataset, eval_dataset
