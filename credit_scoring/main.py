import argparse
import os

import pandas as pd

from model import CreditScoringModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Credit scoring example')
    parser.add_argument('--input-path', type=str, default='/input',
                        help='input dataset path')
    parser.add_argument('--output-path', type=str, default='/output',
                        help='output files path')
    args = parser.parse_args()

    # Get historic loan data
    loan_data_path = os.path.join(args.input_path, "loan_features/table.parquet")
    loans = pd.read_parquet(loan_data_path)

    # Create model
    model = CreditScoringModel(args.output_path)

    # Train model (using Redshift for zipcode and credit history features)
    if not model.is_model_trained():
        model.train(loans)

    # Make online prediction (using DynamoDB for retrieving online features)
    # loan_request = {
    #     "zipcode": [76104],
    #     "dob_ssn": ["19630621_4278"],
    #     "person_age": [133],
    #     "person_income": [59000],
    #     "person_home_ownership": ["RENT"],
    #     "person_emp_length": [123.0],
    #     "loan_intent": ["PERSONAL"],
    #     "loan_amnt": [35000],
    #     "loan_int_rate": [16.02],
    # }
    #
    # result = model.predict(loan_request)
    #
    # if result == 0:
    #     print("Loan approved!")
    # elif result == 1:
    #     print("Loan rejected!")
