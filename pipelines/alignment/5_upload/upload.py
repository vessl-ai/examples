import argparse
from pathlib import Path

import vessl
import vessl.volume


def main(path: Path, volume: str):
    for file_path in path.glob("*"):
        relative_path = str(file_path.relative_to(path))

        source_ref = vessl.volume.parse_volume_url(
            str(file_path), raise_if_not_exists=True
        )
        dest_ref = vessl.volume.parse_volume_url(volume + f"/{relative_path}")

        vessl.volume.copy_volume_file_v2(source_ref, dest_ref)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--volume", type=str, required=True)
    args = parser.parse_args()

    main(args.path, args.model_repo)
