import argparse
import sys
import pandas as pd
import tensorflow as tf

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
    parser.add_argument('--SEED', type=int, default=2023,
                        help='Random Seed')
    parser.add_argument('--num-epochs', type=int, default=1,
                        help="number of training epoch")
    parser.add_argument('--batch-size', type=int, default=128,
                        help="number of batch size")
    parser.add_argument('--lr', type=float, default=0.0005,
                        help="learning rate")
    args = parser.parse_args(args=[])

    env_info()

    # Set model config
    config = {
        "MAXLEN": 50,
        "NUM_BLOCKS": 2,      # NUMBER OF TRANSFORMER BLOCKS
        "HIDDEN_UNITS": 100,  # NUMBER OF UNITS IN THE ATTENTION CALCULATION
        "NUM_HEADS ": 1,      # NUMBER OF ATTENTION HEADS
        "DROPOUT_RATE": 0.2,  # DROPOUT RATE
        "L2_EMB": 0.0,        # L2 REGULARIZATION COEFFICIENT
        "NUM_NEG_TEST": 100,  # NUMBER OF NEGATIVE EXAMPLES PER POSITIVE EXAMPLE
        "TopK" :  10
    }

    # Update hyperparameters for the record to VESSL experiment
    vessl.hp.lr = args.lr
    vessl.hp.batch_size = args.batch_size
    vessl.hp.num_epochs = args.num_epochs
    vessl.hp.SEED = args.SEED
    vessl.hp.update()

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

    # Print useful statistics by recommenders
    num_steps = int(len(rec_data.user_train) / args.batch_size)
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
        batch_size=args.batch_size,
        maxlen=config.get("MAXLEN"),
        n_workers=2
    )

    # Get model with config
    model = get_model(rec_data, config)

    with Timer() as train_time:
        t_test = model.train(
            rec_data,
            sampler,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_epoch=4,
            save_path=args.output_path,
            evaluate=True
        )

    # Print sample input -> next item prediction
    sample_input = np.random.randint(rec_data.itemnum, size=5) + 1
    predictions = -1 * model.predict_next(input=sample_input)
    rec_items = predictions.argsort()[:config["TopK"]] + 1

    print("top{} item recommendataions are {}".format(config["TopK"], rec_items))



