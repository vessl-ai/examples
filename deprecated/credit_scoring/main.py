import argparse
import os

import pandas as pd
from model import CreditScoringModel, MyFeast

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit scoring example")
    parser.add_argument(
        "--feature-repo-path",
        type=str,
        default="/feature_repo",
        help="Feast feature repo path",
    )
    parser.add_argument(
        "--input-path", type=str, default="/input", help="input dataset path"
    )
    parser.add_argument(
        "--output-path", type=str, default="/output", help="output files path"
    )
    args = parser.parse_args()

    # Get historic loan datapyth
    loan_data_path = os.path.join(args.input_path, "loan_features/table.parquet")
    loans = pd.read_parquet(loan_data_path)

    # Create model
    fs = MyFeast(args.feature_repo_path)
    model = CreditScoringModel(args.output_path, fs)

    # Train model (using Redshift for zipcode and credit history features)
    model.train(loans)
