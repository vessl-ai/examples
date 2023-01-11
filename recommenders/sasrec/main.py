import argparse
import sys
import pandas as pd
import tensorflow as tf
import vessl
from recommenders.utils.timer import Timer
from recommenders.datasets.split_utils import filter_k_core

# Sampler for sequential prediction
from recommenders.models.sasrec.sampler import WarpSampler
from recommenders.models.sasrec.util import SASRecDataSet
from model import *
tf.get_logger().setLevel('ERROR')


def env_info():
    print("System version: {}".format(sys.version))
    print("tensorflow version : {}".format(tf.__version__))
    print(tf.config.list_physical_devices('GPU'))
    return


MAXLEN = 50  # MAXIMUM SEQUENCE LENGTH FOR EACH USER
NUM_BLOCKS = 2  # NUMBER OF TRANSFORMER BLOCKS
HIDDEN_UNITS = 100  # NUMBER OF UNITS IN THE ATTENTION CALCULATION
NUM_HEADS = 1  # NUMBER OF ATTENTION HEADS
DROPOUT_RATE = 0.2  # DROPOUT RATE
L2_EMB = 0.0  # L2 REGULARIZATION COEFFICIENT
NUM_NEG_TEST = 100  # NUMBER OF NEGATIVE EXAMPLES PER POSITIVE EXAMPLE

# INPUT_DATA  = "groceries.csv"
# INPUT_DATA_AFTER =   "groceries_preprocessed.txt"

INPUT_DATA = "amazon-beauty.csv"
INPUT_DATA_AFTER = "ratings_Beauty_preprocessed.txt"
OUTPUT_DIR = "model_checkpoints"


def load_data(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    raw_data = pd.read_csv(data_path)
    return raw_data


def get_model(model_name, dataset):
    if model_name == 'sasrec':
        model = SASREC_Vessl(item_num=dataset.itemnum,  # should be changed by data
                             seq_max_len=MAXLEN,
                             num_blocks=NUM_BLOCKS,
                             embedding_dim=HIDDEN_UNITS,
                             attention_dim=HIDDEN_UNITS,
                             attention_num_heads=NUM_HEADS,
                             dropout_rate=DROPOUT_RATE,
                             conv_dims=[100, 100],
                             l2_reg=L2_EMB,
                             num_neg_test=NUM_NEG_TEST
                             )
    elif model_name == "ssept":
        model = SSEPT_Vessl(item_num=dataset.itemnum,  # should be changed by data
                            user_num=dataset.usernum,
                            seq_max_len=MAXLEN,
                            num_blocks=NUM_BLOCKS,
                            # embedding_dim=hidden_units,  # optional
                            user_embedding_dim=10,
                            item_embedding_dim=HIDDEN_UNITS,
                            attention_dim=HIDDEN_UNITS,
                            attention_num_heads=NUM_HEADS,
                            dropout_rate=DROPOUT_RATE,
                            conv_dims=[110, 110],
                            l2_reg=L2_EMB,
                            num_neg_test=NUM_NEG_TEST
                            )
    else:
        raise ValueError(f"Model-{model_name} not found or inappropriate model name")

    return model


if __name__ == '__main__':

    vessl.init()

    # arguments parsing
    parser = argparse.ArgumentParser(description='Pytorch recommender SASREC Example')

    parser.add_argument('--input-path', type=str, default='/input',
                        help='input dataset path')
    parser.add_argument('--evaluate', action='store_true', default=True,
                        help='evaluate during training')

    """parameter that user can choose"""
    parser.add_argument('--SEED', type=int, default=2023,
                        help='Random Seed')
    # training
    parser.add_argument('--num-epochs', type=int, default=20,
                        help="number of training epoch")
    parser.add_argument('--batch-size', type=int, default=128,
                        help="number of batch size")
    parser.add_argument('--lr', type=float, default=0.0005,
                        help="learning rate")
    parser.add_argument('--model-name', type=str, default='sasrec',
                        help="sasrec or ssept")
    args = parser.parse_args(args=[])

    env_info()

    # init argument is also okay.
    vessl.hp.lr = args.lr
    vessl.hp.batch_size = args.batch_size
    vessl.hp.num_epochs = args.num_epochs
    vessl.hp.SEED = args.SEED
    vessl.model_name = args.model_name
    vessl.hp.update()

    df = load_data(args.input_path, INPUT_DATA)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.rename(columns={'UserId': 'userID', 'ProductId': 'itemID'}, inplace=True)

    # df = df[df['Rating'] > 3.5]  #  Not in original paper
    df = df.sort_values(by=["userID", "Timestamp"]).reset_index().drop(columns=['index', 'Timestamp', 'Rating'])
    df = filter_k_core(df, 5)
    item_hashing = {item: idx + 1 for idx, item in enumerate(df.loc[:, 'itemID'].unique())}
    user_hashing = {user: idx + 1 for idx, user in enumerate(df.loc[:, 'userID'].unique())}
    df["itemID"] = df["itemID"].apply(lambda x: item_hashing[x])
    df["userID"] = df["userID"].apply(lambda x: user_hashing[x])

    df.to_csv(os.path.join(args.input_path, INPUT_DATA_AFTER), index=False, header=False, sep="\t")

    # recsystem dataset generation for training
    rec_data = SASRecDataSet(filename=str(os.path.join(args.input_path, INPUT_DATA_AFTER)), col_sep="\t")
    rec_data.split()

    # some statistics by recommenders
    num_steps = int(len(rec_data.user_train) / args.batch_size)
    cc = 0.0
    for u in rec_data.user_train:
        cc += len(rec_data.user_train[u])
    print('%g Users and %g items' % (rec_data.usernum, rec_data.itemnum))
    print('average sequence length: %.2f' % (cc / len(rec_data.user_train)))

    # the sampler creates negative samples from the training data for each batch
    sampler = WarpSampler(rec_data.user_train, rec_data.usernum, rec_data.itemnum, batch_size=args.batch_size,
                          maxlen=MAXLEN, n_workers=2)

    # get model
    model = get_model(args.model_name, rec_data)

    # training (vessl logging is not implemented)
    with Timer() as train_time:
        t_test = model.train(rec_data, sampler, num_epochs=args.num_epochs, batch_size=args.batch_size,
                             lr=args.lr, val_epoch=4, save_path=OUTPUT_DIR, evaluate=True)

    # print sample input -> next item prediction
    sample_input = np.random.randint(rec_data.itemnum, size=5) + 1
    next_item_predict = model.predict_next(input=sample_input)
