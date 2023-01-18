import argparse
import sys
import os
import vessl
import tensorflow as tf

from logger import *
from model import *

from recommenders.utils.timer import Timer
from recommenders.datasets.split_utils import filter_k_core
from recommenders.models.sasrec.sampler import WarpSampler
from recommenders.models.sasrec.util import SASRecDataSet

tf.get_logger().setLevel('ERROR')
vessl.init()

def env_info():
    print("System version: {}".format(sys.version))
    print("tensorflow version : {}".format(tf.__version__))
    print(tf.config.list_physical_devices('GPU'))
    return

def load_data(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    raw_data = pd.read_csv(data_path)
    return raw_data

def get_model(dataset, model_config: dict):
    return SASREC_Vessl(
        item_num=dataset.itemnum,  # should be changed according to data
        seq_max_len=model_config.get("MAXLEN"),
        num_blocks=model_config.get("NUM_BLOCKS"),
        embedding_dim=model_config.get("HIDDEN_UNITS"),
        attention_dim=model_config.get("HIDDEN_UNITS"),
        attention_num_heads=model_config.get("NUM_HEADS"),
        dropout_rate=model_config.get("DROPOUT_RATE"),
        conv_dims=[100, 100],
        l2_reg=model_config.get("L2_EMB"),
        num_neg_test=model_config.get("NUM_NEG_TEST"),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recommender SASREC Example')
    parser.add_argument('--input-path', type=str, default='/input',
                        help='input dataset path')
    parser.add_argument('--output-path', type=str, default='/output',
                        help='output dataset path')
    parser.add_argument('--evaluate', action='store_true', default=True,
                        help='evaluate during training')
    args = parser.parse_args()

    env_info()

    # Set model config
    config = {
        "MAXLEN": 50,
        "NUM_BLOCKS": 2,  # NUMBER OF TRANSFORMER BLOCKS
        "HIDDEN_UNITS": 100,  # NUMBER OF UNITS IN THE ATTENTION CALCULATION
        "NUM_HEADS": 1,  # NUMBER OF ATTENTION HEADS
        "DROPOUT_RATE": 0.2,  # DROPOUT RATE
        "L2_EMB": 0.0,  # L2 REGULARIZATION COEFFICIENT
        "NUM_NEG_TEST": 100,  # NUMBER OF NEGATIVE EXAMPLES PER POSITIVE EXAMPLE
    }

    # Set hyperparameters from environment variables
    lr = float(os.environ.get('lr', 0.0005))
    batch_size = int(os.environ.get('batch_size', 64))
    num_epochs = int(os.environ.get('num_epochs', 20))

    # Load data from VESSL dataset
    df = load_data(args.input_path + '/train', "amazon-beauty.csv")

    # Data preprocessing
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.rename(columns={'UserId': 'userID', 'ProductId': 'itemID'}, inplace=True)

    df = df.sort_values(by=["userID", "Timestamp"]).reset_index().drop(
        columns=['index', 'Timestamp', 'Rating'])
    df = filter_k_core(df, 5)
    item_hashing = {item: idx + 1
                    for idx, item in enumerate(df.loc[:, 'itemID'].unique())}
    user_hashing = {user: idx + 1
                    for idx, user in enumerate(df.loc[:, 'userID'].unique())}
    df["itemID"] = df["itemID"].apply(lambda x: item_hashing[x])
    df["userID"] = df["userID"].apply(lambda x: user_hashing[x])

    preprocessed_input_data_path = os.path.join(args.input_path, "ratings_Beauty_preprocessed.txt")
    df.to_csv(preprocessed_input_data_path, index=False, header=False, sep="\t")

    # Generate recsystem dataset for training
    rec_data = SASRecDataSet(
        filename=preprocessed_input_data_path,
        col_sep="\t"
    )
    rec_data.split()

    # Get model with config
    model = get_model(rec_data, config)

    # Print useful statistics by recommenders
    num_steps = int(len(rec_data.user_train) / batch_size)
    cc = 0.0
    for u in rec_data.user_train:
        cc += len(rec_data.user_train[u])
    print('%g Users and %g items' % (rec_data.usernum, rec_data.itemnum))
    print('average sequence length: %.2f' % (cc / len(rec_data.user_train)))

    # Create negative samples from the training data for each batch
    sampler = WarpSampler(
        rec_data.user_train,
        rec_data.usernum,
        rec_data.itemnum,
        batch_size=batch_size,
        maxlen=config.get("MAXLEN"),
        n_workers=2
    )

    print("train model")
    with Timer() as train_time:
        t_test = model.train(
            rec_data,
            sampler,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            val_epoch=4,
            save_path=args.output_path,
            evaluate=True
        )

    # Print sample input -> top10 next item prediction
    sample_input = np.random.randint(rec_data.itemnum, size=5) + 1
    predictions = -1 * model.predict_next(input=sample_input)
    rec_items = predictions.argsort()[:10]
    result = {k: v for k, v in zip(rec_items + 1, -1 * predictions[rec_items])}

    print("Random sample input :{}".format(sample_input))
    print('Recommended item numbers and their similarity scores(not normalized) for random sample input')
    for key, value in result.items():
        print(key, ":", value)


