import json
import os

import vessl
import matplotlib.pyplot as plt
from pathlib import Path

import feast
import joblib
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import LearningCurveDisplay


class MyFeast:
    feast_features = [
        "zipcode_features:city",
        "zipcode_features:state",
        "zipcode_features:location_type",
        "zipcode_features:tax_returns_filed",
        "zipcode_features:population",
        "zipcode_features:total_wages",
        "credit_history:credit_card_due",
        "credit_history:mortgage_due",
        "credit_history:student_loan_due",
        "credit_history:vehicle_loan_due",
        "credit_history:hard_pulls",
        "credit_history:missed_payments_2y",
        "credit_history:missed_payments_1y",
        "credit_history:missed_payments_6m",
        "credit_history:bankruptcies",
    ]

    target = "loan_status"

    def __init__(self):
        # Set up feature store
        self.fs = feast.FeatureStore(repo_path="feature_repo")

    def get_training_features(self, loans):
        training_df = self.fs.get_historical_features(
            entity_df=loans, features=self.feast_features
        ).to_df()

        train_X = training_df[
            training_df.columns.drop(self.target)
                .drop("event_timestamp")
                .drop("created_timestamp")
                .drop("loan_id")
                .drop("zipcode")
                .drop("dob_ssn")
        ]
        train_X = train_X.reindex(sorted(train_X.columns), axis=1)
        train_Y = training_df.loc[:, self.target]

        return training_df, train_X, train_Y

    def get_online_features_from_feast(self, request):
        print('get_online_features_from_feast is called')
        zipcode = request["zipcode"][0]
        dob_ssn = request["dob_ssn"][0]

        return self.fs.get_online_features(
            entity_rows=[{"zipcode": zipcode, "dob_ssn": dob_ssn}],
            features=self.feast_features,
        ).to_dict()


class CreditScoringModel:
    categorical_features = [
        "person_home_ownership",
        "loan_intent",
        "city",
        "state",
        "location_type",
    ]

    model_filename = "model.bin"
    encoder_filename = "encoder.bin"

    def __init__(self, output_path, fs=None):
        # Load model
        if Path(self.model_filename).exists():
            self.classifier = joblib.load(self.model_filename)
        else:
            self.classifier = tree.DecisionTreeClassifier()

        # Load ordinal encoder
        if Path(self.encoder_filename).exists():
            self.encoder = joblib.load(self.encoder_filename)
        else:
            self.encoder = OrdinalEncoder()

        # # Set up feature store
        # self.fs = feast.FeatureStore(repo_path="feature_repo")
        print("fs:", fs)
        if fs is not None:
            self.fs = fs

        self.output_path = output_path

    def set_feast(self, fs):
        self.fs = fs

    def train(self, loans):
        training_df, train_X, train_Y = self.fs.get_training_features(loans)

        self._fit_ordinal_encoder(training_df)
        self._apply_ordinal_encoding(training_df)

        self.classifier.fit(train_X[sorted(train_X)], train_Y)
        fig, ax = plt.subplots(1, figsize=(10, 10))
        common_params = {
            "X": train_X[sorted(train_X)],
            "y": train_Y,
            "score_type": "both",
            "n_jobs": 4,
            "line_kw": {"marker": "o"},
            "std_display_style": "fill_between",
            "score_name": "Accuracy",
        }
        LearningCurveDisplay.from_estimator(self.classifier, **common_params, ax=ax)
        handles, label = ax.get_legend_handles_labels()
        ax.legend(handles[:2], ["Training Score", "Test Score"])
        title = f"Learning Curve for {self.classifier.__class__.__name__}"
        ax.set_title(title)
        file_path = os.path.join(self.output_path, "learning_curve.png")
        plt.savefig(file_path)

        vessl.log({
            "log-image": [vessl.Image(data=file_path, caption=title)],
        })

        model_path = os.path.join(self.output_path, self.model_filename)
        joblib.dump(self.classifier, model_path)

    def _fit_ordinal_encoder(self, requests):
        self.encoder.fit(requests[self.categorical_features])
        encoder_path = os.path.join(self.output_path, self.encoder_filename)
        joblib.dump(self.encoder, encoder_path)

    def _apply_ordinal_encoding(self, requests):
        requests[self.categorical_features] = self.encoder.transform(
            requests[self.categorical_features]
        )

    def predict(self, features_df):
        print("predict is called!")
        # Apply ordinal encoding to categorical features
        self._apply_ordinal_encoding(features_df)

        # Sort columns
        features_df = features_df.reindex(sorted(features_df.columns), axis=1)

        # Drop unnecessary columns
        features_df = features_df[features_df.columns.drop("zipcode").drop("dob_ssn")]

        # Make prediction
        print("features_df:", features_df, type(features_df))
        features_df["prediction"] = self.classifier.predict(features_df)

        # return result of credit scoring
        return features_df["prediction"].iloc[0]


class MyRunner(vessl.RunnerBase):
    @staticmethod
    def load_model(props, artifacts):
        print('load_model started')
        model = CreditScoringModel(output_path="/output")
        print('model:', model)
        return model

    @staticmethod
    def preprocess_data(data):
        print('preprocess_data started')
        request = json.loads(data.decode('utf-8'))

        # Get online features from Feast
        fs = MyFeast()
        feature_vector = fs.get_online_features_from_feast(request)
        print('feature_vector:', feature_vector)

        # Join features to request features
        features = request.copy()
        features.update(feature_vector)
        features_df = pd.DataFrame.from_dict(features)
        return {
            "df": features_df,
            "fs": fs,
        }

    @staticmethod
    def predict(model, data):
        model.set_feast(data["fs"])
        return model.predict(data["df"])

    @staticmethod
    def postprocess_data(data):
        print("post process data:", data)
        return data


if __name__ == '__main__':
    vessl.configure()

    model_repository_name = "credit-scoring"

    # vessl.create_model_repository(name=model_repository_name)
    #
    # model_repository = vessl.read_model_repository(
    #     repository_name=model_repository_name,
    # )

    vessl.register_model(
        repository_name=model_repository_name,
        model_number=2,
        runner_cls=MyRunner,
        requirements=["feast"]
    )
