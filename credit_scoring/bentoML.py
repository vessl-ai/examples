"""
Script for bentoML
"""

import argparse
import os
import pandas as pd

import bentoml
from bentoml.io import PandasDataFrame, Text , NumpyNdarray
import bentoml.io
from model import CreditScoringModel, MyFeast

from matplotlib import pyplot as plt

from model import CreditScoringModel, MyFeast



if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="bentoML")
    parser.add_argument('--output-path', type=str, default='output',
                        help='output files path')

    args = parser.parse_args()


    model = CreditScoringModel(args.output_path, fs = None)

    # Save model to bento local model store
    bentoml.sklearn.save_model("credit_scoring_model", model)
    ## path는 못넘김. #이 tag에 :awdalkwegja 등과 creation time 이 걸림.


    print("script ends")









