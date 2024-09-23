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


def sharegpt_to_chatml(sample):
    conversation = sample["conversations"]

    system_content = [c for c in conversation if c["from"] == "system"]
    system = system_content[0]["value"] if system_content else ""
    user = [c for c in conversation if c["from"] == "human"][0]["value"]
    assistant = [c for c in conversation if c["from"] == "gpt"][0]["value"]

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def meltal_health_to_chatml(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a mental health support assistant. Respond empathetically to users, offer gentle encouragement, and suggest positive coping strategies. Do not diagnose or replace professional care. If a user expresses severe distress or danger, advise them to seek immediate professional help. Maintain a warm, supportive tone in all interactions.",
            },
            {"role": "user", "content": sample["Context"]},
            {"role": "assistant", "content": sample["Response"]},
        ]
    }


def create_dataset(data_args: DatasetArguments):
    try:
        dataset = load_dataset(data_args.dataset_name)
    except DatasetGenerationError:
        dataset = load_from_disk(data_args.dataset_name)

    # inappropriately hardcoded
    column_names = dataset.column_names["train"]
    if set(column_names) >= {"input", "instruction", "output"}:
        # alpaca
        dataset = dataset.map(alpaca_to_chatml)
        dataset = dataset.remove_columns(
            [col for col in column_names if col != "messages"]
        )
    elif "conversations" in column_names and (
        set(dataset["train"][0]["conversations"][0].keys())
        >= {"from", "value", "weight"}
    ):
        # sharegpt
        dataset = dataset.map(sharegpt_to_chatml)
        dataset = dataset.remove_columns(
            [col for col in column_names if col != "messages"]
        )
    elif set(column_names) >= {"Context", "Response"}:
        # mental health care
        dataset = dataset.map(meltal_health_to_chatml)
        dataset = dataset.remove_columns(
            [col for col in column_names if col != "messages"]
        )

    train_dataset = dataset["train"]
    if "val" in dataset.keys():
        eval_dataset = dataset["val"]
    else:
        eval_dataset = None

    return train_dataset, eval_dataset
