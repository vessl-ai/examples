import argparse
import json
from pathlib import Path


def update_dataset_info(path: str):
    dataset_path = Path(path)

    # NOTE(sanghyuk): is there any elegant way to resolve the path to the top level dir?
    root_path = Path(__file__).resolve().parents[1]

    with open(root_path.joinpath("data/dataset_info.json")) as fp:
        dataset_info = json.load(fp)

    with open(dataset_path.joinpath("dataset_info.json")) as fp:
        new_dataset_info = json.load(fp)
        for key, info in new_dataset_info.items():
            file_name = str(dataset_path.joinpath(info["file_name"]))
            info["file_name"] = file_name
            new_dataset_info[key] = info

    common_keys = set(dataset_info.keys()).intersection(new_dataset_info.keys())
    if common_keys:
        print("Following datasets appear in both llama-factory dataset info and custom dataset info.")
        print("The custom dataset info will be selected.")
        print("\n".join([f"- {key}" for key in common_keys]))

    with open(root_path.joinpath("data/dataset_info.json"), "w") as fp:
        dataset_info.update(new_dataset_info)
        json.dump(dataset_info, fp, indent=2)


def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    main(args.path)
