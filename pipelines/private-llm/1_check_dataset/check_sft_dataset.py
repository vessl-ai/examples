import argparse

import vessl
from datasets import Value, load_dataset, load_from_disk

SFT_FORMAT_MAPPING = {
    "chatml": [
        {
            "content": Value(dtype="string", id=None),
            "role": Value(dtype="string", id=None),
        }
    ],
    "instruction": {
        "completion": Value(dtype="string", id=None),
        "prompt": Value(dtype="string", id=None),
    },
}


def main(dataset_name_or_path: str):
    try:
        dataset = load_from_disk(dataset_name_or_path)["train"]
    except FileNotFoundError:
        dataset = load_dataset(dataset_name_or_path)["train"]

    if "messages" in dataset.features:
        if dataset.features["messages"] == SFT_FORMAT_MAPPING["chatml"]:
            print(f"The dataset ({dataset_name_or_path}) is a SFT dataset.")
            vessl.update_context_variables({"IS_SFT_DATASET": "YES"})
    elif "conversations" in dataset.features:
        if dataset.features["conversations"] == SFT_FORMAT_MAPPING["chatml"]:
            print(f"The dataset ({dataset_name_or_path}) is a SFT dataset.")
            vessl.update_context_variables({"IS_SFT_DATASET": "YES"})
    elif dataset.features.items() >= SFT_FORMAT_MAPPING["instruction"].items():
        print(f"The dataset ({dataset_name_or_path}) is a SFT dataset.")
        vessl.update_context_variables({"IS_SFT_DATASET": "YES"})
    else:
        print(f"The dataset ({dataset_name_or_path}) is not a SFT dataset.")
        vessl.update_context_variables({"IS_SFT_DATASET": "NO"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name-or-path", type=str, required=True)
    args = parser.parse_args()

    main(dataset_name_or_path=args.dataset_name_or_path)
