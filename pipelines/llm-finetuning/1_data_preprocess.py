import os

import datasets
import fitz
from datasets.exceptions import DataFilesNotFoundError

DATASET_PATH = os.environ.get("DATASET_PATH", "/input")
HF_TOKEN = os.environ.get("HF_TOKEN")


def main():
    try:
        insurance_policies_data = datasets.load_dataset(
            "vessl/insurance-policies", split="train"
        )
    except DataFilesNotFoundError:
        insurance_policies_data = datasets.Dataset.from_dict(
            {"filename": [], "insurance_name": [], "text": []}
        )

    new_file_count = 0
    existing_filenames = set(insurance_policies_data["filename"])
    for _, _, filenames in os.walk(DATASET_PATH):
        for filename in filenames:
            if not filename.endswith(".pdf"):
                continue

            if filename in existing_filenames:
                continue

            text = ""
            with fitz.open(os.path.join(DATASET_PATH, filename)) as pdf:
                for page in pdf:
                    text += page.get_text()

            if text == "":
                print(f"Skipping empty file: {filename}")
                continue

            # Assume filename is the name of insurance name with extension
            insurance_name = os.path.splitext(filename)[0]
            insurance_policies_data = insurance_policies_data.add_item(
                {
                    "filename": filename,
                    "insurance_name": insurance_name,
                    "text": text.replace("\n", " "),
                },
            )
            new_file_count += 1

    if new_file_count > 0:
        insurance_policies_data.push_to_hub("vessl/insurance-policies", token=HF_TOKEN)
        if new_file_count == 1:
            print("Pushed 1 new file to the dataset.")
        else:
            print(f"Pushed {new_file_count} new files to the dataset.")
    else:
        print("No new files found.")


if __name__ == "__main__":
    main()
