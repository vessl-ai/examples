import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from savvihub.keras import SavviHubCallback
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def load_data(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    raw_data = pd.read_csv(data_path, dtype=np.float32)
    return raw_data


def preprocess(raw_data):
    label = raw_data["label"].to_numpy()
    data = raw_data.drop(labels=["label"], axis=1)
    data = data / 255.0
    data = data.values.reshape(-1, 28, 28, 1)
    return label, data


def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])


def save(model, path):
    if not os.path.exists(path):
        print(f" [*] Make directories : {path}")
        os.makedirs(path)
    artifact_path = os.path.join(path, "my_model")
    model.save(artifact_path)
    print(f" [*] Saved model in : {artifact_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras MNIST Example')
    parser.add_argument('--input-path', type=str, default='/input',
                        help='input dataset path')
    parser.add_argument('--output-path', type=str, default='/output',
                        help='output files path')
    parser.add_argument('--checkpoint-path', type=str, default='/output/checkpoint',
                        help='checkpoint file path')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    print(f'=> Available GPUs: {get_available_gpus()}')

    use_mount_dataset = False
    if os.path.exists(os.path.join(args.input_path, "train.csv")) and \
            os.path.exists(os.path.join(args.input_path, 'test.csv')):
        use_mount_dataset = True

    if use_mount_dataset:
        print('=> Mount dataset found!')
        train_df = load_data(args.input_path, "train.csv")
        test_df = load_data(args.input_path, 'test.csv')
        y_train, x_train = preprocess(train_df)
        y_test, x_test = preprocess(test_df)
    else:
        print('=> Mount dataset not found! Use keras dataset instead.')
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

    random_seed = 7
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=random_seed)

    model = create_model()
    print(model.summary())

    # Load checkpoint if exists
    checkpoint_file_path = os.path.join(args.checkpoint_path, 'checkpoints.hdf5')
    if os.path.exists(args.checkpoint_path) and os.path.isfile(checkpoint_file_path):
        print(f"=> Loading checkpoint '{checkpoint_file_path}' ...")
        model.load_weights(checkpoint_file_path)
    else:
        if not os.path.exists(args.checkpoint_path):
            print(f" [*] Make directories : {args.checkpoint_path}")
            os.makedirs(args.checkpoint_path)

    # Compile model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Prepare checkpoint
    checkpoint_file_path = os.path.join(args.checkpoint_path, 'checkpoints.hdf5')
    checkpoint_callback = ModelCheckpoint(
        checkpoint_file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              validation_data=(x_val, y_val),
              epochs=args.epochs,
              callbacks=[
                  SavviHubCallback(data_type='image', validation_data=(x_val, y_val), num_images=5),
                  checkpoint_callback,
              ])

    model.evaluate(x_test, y_test, verbose=2)

    if args.save_model:
        save(model, args.output_path)
