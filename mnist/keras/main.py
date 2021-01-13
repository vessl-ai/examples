import argparse
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from savvihub.keras import SavviHubCallback

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau

from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def load_data(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    raw_data = pd.read_csv(data_path, dtype=np.float32)
    return raw_data


def preprocess(raw_data):
    label = raw_data["label"]
    data = raw_data.drop(labels=["label"], axis=1)
    data = data / 255.0
    data = data.values.reshape(-1, 28, 28, 1)
    return label, data


def create_model():
    return Sequential(
        [
            Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)),
            Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax")
        ]
    )


def one_hot_to_class(label, axis):
    return np.argmax(label, axis)


def save(model, path):
    if not os.path.exists(path):
        print(f" [*] Make directories : {path}")
        os.makedirs(path)
    artifact_path = os.path.join(path, "my_model")
    model.save(artifact_path)
    print(f" [*] Saved model in : {artifact_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras MNIST Example')
    parser.add_argument('--input-path', type=str, default='/input', help='input dataset path')
    parser.add_argument('--output-path', type=str, default='/output', help='output files path')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    print(get_available_gpus())

    train_df = load_data(args.input_path, "train.csv")
    test_df = load_data(args.input_path, 'test.csv')

    train_label, train_data = preprocess(train_df)
    test_label, test_data = preprocess(test_df)

    print(f'The shape of train data: {train_data.shape}')
    print(f'The shape of test data: {test_data.shape}')

    train_label = to_categorical(train_label, num_classes=10)

    random_seed = 7
    train_data, val_data, train_label, val_label = train_test_split(
        train_data, train_label, test_size=0.1, random_state=random_seed)

    model = create_model()
    print(model.summary())

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

    history = model.fit(train_data, train_label,
                        batch_size=args.batch_size, epochs=args.epochs,
                        validation_data=(val_data, val_label), verbose=2,
                        callbacks=[SavviHubCallback()])

    test_pred = model.predict(test_data)
    test_pred_class = one_hot_to_class(test_pred, 1)
    accuracy = accuracy_score(test_label, test_pred_class)
    print('Test accuracy: {:.2f}%'.format(accuracy * 100))

    if args.save_model:
        save(model, args.output_path)
