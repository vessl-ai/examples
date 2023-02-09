
import bentoml
from bentoml.io import PandasDataFrame, Text , NumpyNdarray
import pandas as pd

# 1. create runner instance
# 2. simple service generation
credit_scoring_runner = bentoml.sklearn.get("credit_scoring_model:latest").to_runner()
svc = bentoml.Service("credit_classifier", runners=[credit_scoring_runner])

@svc.api(input=PandasDataFrame(), output=NumpyNdarray())
def scoring(input_df: pd.DataFrame) -> str :

    tmp = credit_scoring_runner.set_feast.run(input_df)
    print("set_feast is done : ", tmp )

    return credit_scoring_runner.predict.run(tmp)

