import argparse

import vessl
from datasets import load_dataset


def main(dataset_name_or_path: str):
    dataset = load_dataset(dataset_name_or_path)
    if all(col in dataset.column_names for col in ["chosen", "rejected"]):
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
