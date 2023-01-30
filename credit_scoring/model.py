import os
from pathlib import Path

import feast
import joblib
import pandas as pd
import sklearn.metrics
from sklearn import tree
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import LearningCurveDisplay

import matplotlib.pyplot as plt


class CreditScoringModel:
    categorical_features = [
        "person_home_ownership",
        "loan_intent",
        "city",
        "state",
        "location_type",
    ]

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
    model_filename = "model.bin"
    encoder_filename = "encoder.bin"

    def __init__(self):
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

        # Set up feature store
        self.fs = feast.FeatureStore(repo_path="feature_repo")

    def train(self, loans):
        train_X, train_Y = self._get_training_features(loans)

        self.classifier.fit(train_X[sorted(train_X)], train_Y)
        print("score:", self.classifier.score(train_X[sorted(train_X)], train_Y))
        # train_size_abs, train_scores, test_scores = learning_curve(self.classifier, train_X[sorted(train_X)], train_Y)
        # for train_size, cv_train_scores, cv_test_scores in zip(train_size_abs, train_scores, test_scores):
        #     print(f"{train_size} samples were used to train the model")
        #     print(f"The average train accuracy is {cv_train_scores.mean():.2f}")
        #     print(f"The average test accuracy is {cv_test_scores.mean():.2f}")

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
        file_path = os.path.join("/output", "learning_curve.png")
        plt.savefig(file_path)

        vessl.log({
            "log-image": [vessl.Image(data=file_path, caption=title)],
        })

        joblib.dump(self.classifier, self.model_filename)

    def _get_training_features(self, loans):
        training_df = self.fs.get_historical_features(
            entity_df=loans, features=self.feast_features
        ).to_df()

        self._fit_ordinal_encoder(training_df)
        self._apply_ordinal_encoding(training_df)

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

        return train_X, train_Y

    def _fit_ordinal_encoder(self, requests):
        self.encoder.fit(requests[self.categorical_features])
        joblib.dump(self.encoder, self.encoder_filename)

    def _apply_ordinal_encoding(self, requests):
        requests[self.categorical_features] = self.encoder.transform(
            requests[self.categorical_features]
        )

    def predict(self, request):
        # Get online features from Feast
        feature_vector = self._get_online_features_from_feast(request)

        # Join features to request features
        features = request.copy()
        features.update(feature_vector)
        features_df = pd.DataFrame.from_dict(features)

        # Apply ordinal encoding to categorical features
        self._apply_ordinal_encoding(features_df)

        # Sort columns
        features_df = features_df.reindex(sorted(features_df.columns), axis=1)

        # Drop unnecessary columns
        features_df = features_df[features_df.columns.drop("zipcode").drop("dob_ssn")]

        # Make prediction
        features_df["prediction"] = self.classifier.predict(features_df)

        # return result of credit scoring
        return features_df["prediction"].iloc[0]

    def _get_online_features_from_feast(self, request):
        zipcode = request["zipcode"][0]
        dob_ssn = request["dob_ssn"][0]

        return self.fs.get_online_features(
            entity_rows=[{"zipcode": zipcode, "dob_ssn": dob_ssn}],
            features=self.feast_features,
        ).to_dict()

    def is_model_trained(self):
        try:
            check_is_fitted(self.classifier, "tree_")
        except NotFittedError:
            return False
        return True
