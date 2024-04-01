import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vessl
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.core import Dense
from vessl.integration.keras import ExperimentCallback

vessl.init()


def preprocess_original_data(input_path):
    df = pd.read_csv(os.path.join(input_path, "creditcard.csv"))

    robust_scaler = RobustScaler()

    df["scaled_amount"] = robust_scaler.fit_transform(
        df["Amount"].values.reshape(-1, 1)
    )
    df["scaled_time"] = robust_scaler.fit_transform(df["Time"].values.reshape(-1, 1))

    df.drop(["Time", "Amount"], axis=1, inplace=True)

    scaled_amount = df["scaled_amount"]
    scaled_time = df["scaled_time"]

    df.drop(["scaled_amount", "scaled_time"], axis=1, inplace=True)
    df.insert(0, "scaled_amount", scaled_amount)
    df.insert(1, "scaled_time", scaled_time)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    X_test, y_test = None, None
    for train_index, test_index in sss.split(X, y):
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

    return X_test.values, y_test.values


def plot_confusion_matrix(
    cm,
    classes,
    output_path,
    filename,
    save_image,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if save_image:
        file_path = os.path.join(output_path, filename)
        plt.savefig(file_path)

        vessl.log(
            {
                "log-image": [vessl.Image(data=file_path, caption=title)],
            }
        )


def save(model, path):
    if not os.path.exists(path):
        print(f" [*] Make directories : {path}")
        os.makedirs(path)
    artifact_path = os.path.join(path, "my_model")
    model.save(artifact_path)
    print(f" [*] Saved model in : {artifact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keras credit card example")
    parser.add_argument(
        "--input-path", type=str, default="/input", help="input dataset path"
    )
    parser.add_argument(
        "--output-path", type=str, default="/output", help="output files path"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For saving the current model",
    )
    parser.add_argument(
        "--save-image", action="store_true", default=False, help="For saving the images"
    )
    args = parser.parse_args()

    epochs = int(os.environ.get("epochs", 25))

    # Read an under sampling dataset from csv file
    X = pd.read_csv(os.path.join(args.input_path, "X.csv"), index_col=0)
    y = pd.read_csv(
        os.path.join(args.input_path, "y.csv"), index_col=0, header=0, squeeze=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_test = X_train.values, X_test.values
    y_train, y_test = y_train.values, y_test.values

    n_inputs = X_train.shape[1]
    under_sample_model = Sequential(
        [
            Dense(n_inputs, input_shape=(n_inputs,), activation="relu"),
            Dense(32, activation="relu"),
            Dense(2, activation="softmax"),
        ]
    )

    print(under_sample_model.summary())

    under_sample_model.compile(
        Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    under_sample_model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=25,
        epochs=20,
        shuffle=True,
        verbose=2,
        callbacks=[ExperimentCallback()],
    )

    under_sample_model.evaluate(X_test, y_test, verbose=2)

    original_X_test, original_y_test = preprocess_original_data(args.input_path)

    under_sample_predictions = under_sample_model.predict(
        original_X_test, batch_size=200, verbose=0
    )
    under_sample_fraud_predictions = np.argmax(under_sample_predictions, axis=1)

    under_sample_cm = confusion_matrix(original_y_test, under_sample_fraud_predictions)
    actual_cm = confusion_matrix(original_y_test, original_y_test)
    labels = ["No Fraud", "Fraud"]

    plot_confusion_matrix(
        under_sample_cm,
        labels,
        output_path=args.output_path,
        filename="test1.png",
        save_image=args.save_image,
        title="Random UnderSample \n Confusion Matrix",
        cmap=plt.cm.Reds,
    )

    plot_confusion_matrix(
        actual_cm,
        labels,
        output_path=args.output_path,
        filename="test2.png",
        save_image=args.save_image,
        title="Confusion Matrix \n (with 100% accuracy)",
        cmap=plt.cm.Greens,
    )

    if args.save_model:
        save(under_sample_model, args.output_path)
