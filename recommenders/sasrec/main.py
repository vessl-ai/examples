import argparse
import sys
import os
import vessl
import numpy as np
import tensorflow as tf
import pandas as pd

from logger import *

from recommenders.utils.timer import Timer
from recommenders.datasets.split_utils import filter_k_core
from recommenders.models.sasrec.sampler import WarpSampler
from recommenders.models.sasrec.util import SASRecDataSet
from recommenders.models.sasrec.model import SASREC
from tqdm import tqdm
from io import BytesIO

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


class SASREC_Vessl(SASREC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, dataset, sampler, **kwargs):
        """
        High level function for model training as well as
        evaluation on the validation and test dataset and
        for logging training and from source code of recommeders

        :param model: model to train
        :param dataset: dataset
        :param sampler: sampler
        :param kwargs:
        :return:
        """

        if kwargs['save_path'] is not None and not os.path.isdir(kwargs['save_path']):
            os.mkdir(kwargs['save_path'])

        vessllogger = VesslLogger()

        num_epochs = kwargs.get("num_epochs", 10)
        batch_size = kwargs.get("batch_size", 128)
        lr = kwargs.get("learning_rate", 0.001)
        val_epoch = kwargs.get("val_epoch", 5)
        evaluate = kwargs.get("evaluate" , True)

        num_steps = int(len(dataset.user_train) / batch_size)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7
        )

        loss_function = self.loss_function

        train_loss = tf.keras.metrics.Mean(name="train_loss")

        train_step_signature = [
            {
                "users": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
                "input_seq": tf.TensorSpec(
                    shape=(None, self.seq_max_len), dtype=tf.int64
                ),
                "positive": tf.TensorSpec(
                    shape=(None, self.seq_max_len), dtype=tf.int64
                ),
                "negative": tf.TensorSpec(
                    shape=(None, self.seq_max_len), dtype=tf.int64
                ),
            },
            tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            with tf.GradientTape() as tape:
                pos_logits, neg_logits, loss_mask = self(inp, training=True)
                loss = loss_function(pos_logits, neg_logits, loss_mask)

            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            train_loss(loss)
            return loss

        T = 0.0
        t0 = Timer()
        t0.start()

        max_ndgc = 0
        t_test = None
        for epoch in range(1, num_epochs + 1):

            train_loss.reset_states()

            for step in tqdm(range(num_steps), total=num_steps, ncols=70, leave=False, unit="b"):
                u, seq, pos, neg = sampler.next_batch()

                inputs, target = self.create_combined_dataset(u, seq, pos, neg)

                loss = train_step(inputs, target)

                vessllogger.log(step + (epoch - 1) * num_steps, 'loss', loss)

            self.save_upload(kwargs['save_path'], epoch)

            if (epoch % val_epoch == 0 or epoch == num_epochs) and evaluate :
                t0.stop()
                t1 = t0.interval
                T += t1
                print("Evaluating...")
                t_test = self.evaluate(dataset)
                t_valid = self.evaluate(dataset)
                print(
                    f"\nepoch: {epoch}, time: {T}, valid (NDCG-10: {t_valid[0]}, HR-10: {t_valid[1]})"
                )
                vessllogger.log(epoch, 'val_NDCG-10', t_valid[0])
                vessllogger.log(epoch, 'val_HR-10', t_valid[1])
                print(
                    f"epoch: {epoch}, time: {T},  test (NDCG-10: {t_test[0]}, HR-10: {t_test[1]})"
                )
                vessllogger.log(epoch, 'test_NDCG-10', t_test[0])
                vessllogger.log(epoch, 'test_HR-10', t_test[1])

                if max_ndgc < t_test[1] and kwargs['save_path'] is not None:
                    max_ndgc = t_test[1]

                    self.save_weights(str(os.path.join(kwargs['save_path'], 'best')))
                    vessl.upload(str(os.path.join(kwargs['save_path'], 'best')))

                t0.start()
        self.load_weights(str(os.path.join(kwargs['save_path'], 'best')))
        return t_test

    def predict_next(self, input):
        # seq generation
        training = False
        seq = np.zeros([self.seq_max_len], dtype=np.int32)
        idx = self.seq_max_len - 1
        idx -= 1
        for i in input[::-1]:
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        input_seq = np.array([seq])
        candidate = np.expand_dims(np.arange(1, self.item_num+1, 1), axis=0)

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)
        seq_embeddings += positional_embeddings
        seq_embeddings *= mask
        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)
        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.embedding_dim],
        )  # (b*s, d)
        candidate_emb = self.item_embedding_layer(candidate)  # (b, s, d)
        candidate_emb = tf.transpose(candidate_emb, perm=[0, 2, 1])  # (b, d, s)

        test_logits = tf.matmul(seq_emb, candidate_emb)
        test_logits = tf.reshape(
            test_logits,
            [tf.shape(input_seq)[0], self.seq_max_len, self.item_num],
        )
        test_logits = test_logits[:, -1, :]  # (1, 101)

        predictions = np.array(test_logits)[0]

        return predictions

    def save_upload(self, save_path, epoch):
        self.save_weights(str(os.path.join(save_path, 'epoch_{}'.format(epoch))))
        self.load_weights(str(os.path.join(save_path, 'epoch_{}'.format(epoch))))


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


class MyRunner(vessl.RunnerBase):
    @staticmethod
    def load_model(props, artifacts):

        try:
            artifacts.load_weights('best')
        except:
            print("no best model weight file exists")

        return artifacts

    @staticmethod
    def preprocess_data(data):
        data = np.array(list(map(int, list(pd.read_csv(BytesIO(data)).columns))))
        return data

    @staticmethod
    def predict(model, data):
        return model.predict_next(data)

    @staticmethod
    def postprocess_data(data):
        predictions = -1 * data
        rec_items = predictions.argsort()[:10]
        result = {k: v for k, v in zip(rec_items + 1, -1 * predictions[rec_items])}

        print('Recommended item numbers and their similarity scores(not normalized)')
        for key, value in result.items():
            print(key, ":", value)

        output_msg = "buy item{}".format(rec_items[0] + 1)
        return output_msg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recommender SASREC Example')
    parser.add_argument('--input-path', type=str, default='/input',
                        help='input dataset path')
    parser.add_argument('--output-path', type=str, default='/output',
                        help='output dataset path')
    parser.add_argument('--evaluate', action='store_true', default=True,
                        help='evaluate during training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'serving', 'both'],
                        help='train or serving or both')
    args = parser.parse_args()

    env_info()

    # Set model config
    config = {
        "MAXLEN": 50,
        "NUM_BLOCKS": 2,      # NUMBER OF TRANSFORMER BLOCKS
        "HIDDEN_UNITS": 100,  # NUMBER OF UNITS IN THE ATTENTION CALCULATION
        "NUM_HEADS": 1,      # NUMBER OF ATTENTION HEADS
        "DROPOUT_RATE": 0.2,  # DROPOUT RATE
        "L2_EMB": 0.0,        # L2 REGULARIZATION COEFFICIENT
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

    # Run train or serving by mode chosen
    if args.mode == 'train' or 'both':
        print("train model")

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

    elif args.mode == 'serving' or 'both':
        print("serve model")
        vessl.configure()

        model_repository_name = "sequential-recsys"

        vessl.create_model_repository(
            name=model_repository_name
        )

        model_repository = vessl.read_model_repository(
            repository_name=model_repository_name,
        )

        vessl.register_model(
            repository_name=model_repository.name,
            model_number=None,
            runner_cls=MyRunner,
            requirements=["recommenders", "vessl"],
            model_instance=model
        )
