import argparse

import bentoml.io
from model import CreditScoringModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bentoML")
    parser.add_argument(
        "--output-path", type=str, default="output", help="output files path"
    )
    args = parser.parse_args()

    model = CreditScoringModel(args.output_path, fs=None)

    # Save model to bento local model store
    bentoml.sklearn.save_model("credit_scoring_model", model)
