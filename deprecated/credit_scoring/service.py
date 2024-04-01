import bentoml
from bentoml.io import JSON, Text
from model import *

# 2. simple service generation
credit_scoring_runner = bentoml.sklearn.get("credit_scoring_model:latest").to_runner()
svc = bentoml.Service("credit_classifier", runners=[credit_scoring_runner])


@svc.api(input=JSON(), output=Text())
def scoring(inputs) -> str:
    model = CreditScoringModel(output_path="ouput")
    request = inputs

    # Get online features from Feast
    fs = MyFeast(repo_path="feature_repo")
    feature_vector = fs.get_online_features_from_feast(request)

    # Join features to request features
    features = request.copy()
    features.update(feature_vector)
    features_df = pd.DataFrame.from_dict(features)

    data = {
        "df": features_df,
        "fs": fs,
    }

    model.set_feast(data["fs"])

    data = model.predict(data["df"])

    if data == 0:
        msg = "Loan approved!"
        print(msg)

    elif data == 1:
        msg = "Loan rejected!"
        print(msg)

    else:
        msg = "Model prediction failed."
        print(msg)

    return msg
