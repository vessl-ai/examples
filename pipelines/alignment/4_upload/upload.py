import argparse
from pathlib import Path

import vessl
from vessl.util.exception import VesslApiException


def main(path: Path, model_repo: str):
    try:
        _ = vessl.read_model_repository(model_repo)
    except VesslApiException:
        vessl.create_model_repository(model_repo)

    model_info = vessl.create_model(model_repo)
    model_number = model_info.number

    for file_path in path.glob("*"):
        relative_path = file_path.relative_to(path)

        vessl.upload_model_volume_file(
            model_repo,
            model_number,
            str(file_path),
            str("/" / relative_path),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--model-repo", type=str, required=True)
    args = parser.parse_args()

    main(args.path, args.model_repo)
