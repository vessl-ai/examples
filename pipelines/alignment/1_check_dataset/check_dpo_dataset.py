import argparse

import vessl
from datasets import Value, load_dataset

PREF_FORMAT_MAPPING = {
    "chosen": Value(dtype="string", id=None),
    "rejected": Value(dtype="string", id=None),
}


def main(dataset_name_or_path: str):
    dataset = load_dataset(dataset_name_or_path)["train"]

    if dataset.features.items() >= PREF_FORMAT_MAPPING.items():
        print(f"The dataset ({dataset_name_or_path}) is a preference dataset.")
        vessl.update_context_variables({"IS_PREF_DATASET": "YES"})
    else:
        print(f"The dataset ({dataset_name_or_path}) is not a preference dataset.")
        vessl.update_context_variables({"IS_PREF_DATASET": "NO"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name-or-path", type=str, required=True)
    args = parser.parse_args()

    main(dataset_name_or_path=args.dataset_name_or_path)
