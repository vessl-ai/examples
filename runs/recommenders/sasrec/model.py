import os
import time
from io import BytesIO

import numpy as np
import pandas as pd
import tensorflow as tf
import vessl
from recommenders.models.sasrec.model import SASREC
from recommenders.utils.timer import Timer
from tabulate import tabulate
from tqdm import tqdm


class VesslLogger:
    """VESSL logger"""

    def __init__(self):
        """Initializer"""
        self._log = {}

    def log(self, step, metric, value):
        """Log metrics. Each metric's log will be stored in the corresponding list.
        Args:
            step (int): step
            metric (str): Metric name.
            value (float): Value.
        """
        if metric not in self._log:
            self._log[metric] = []
        self._log[metric].append(value)
        vessl.log(
            step=step,
            payload={
                metric: value,
            },
        )

    def get_log(self):
        """Getter
        Returns:
            dict: Log metrics.
        """
        return self._log


class SASREC_Vessl(SASREC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, dataset, sampler, **kwargs):
        """
        High level function for model training as well as
        evaluation on the validation and test dataset and
        for logging training and from source code of recommenders

        :param dataset: dataset
        :param sampler: sampler
        :param kwargs:
        :return:
        """

        if kwargs["save_path"] is not None and not os.path.isdir(kwargs["save_path"]):
            os.mkdir(kwargs["save_path"])

        # vessllogger = VesslLogger()

        num_epochs = kwargs.get("num_epochs", 10)
        batch_size = kwargs.get("batch_size", 128)
        lr = kwargs.get("learning_rate", 0.001)
        val_epoch = kwargs.get("val_epoch", 5)
        evaluate = kwargs.get("evaluate", True)
        wandb_log = kwargs.get("wandb_log", False)

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

            for step in tqdm(
                range(num_steps), total=num_steps, ncols=70, leave=False, unit="b"
            ):
                u, seq, pos, neg = sampler.next_batch()

                inputs, target = self.create_combined_dataset(u, seq, pos, neg)

                loss = train_step(inputs, target)

                if wandb_log:
                    import wandb
                    wandb.log(step=step+(epoch-1)*num_steps, data={"loss": loss})

                # vessllogger.log(step + (epoch - 1) * num_steps, "loss", loss)

            self.save_upload(kwargs["save_path"], epoch)

            if (epoch % val_epoch == 0 or epoch == num_epochs) and evaluate:
                t0.stop()
                t1 = t0.interval
                T += t1
                print("Evaluating...")
                t_test = self.evaluate(dataset)
                t_valid = self.evaluate(dataset)
                print(
                    f"\nepoch: {epoch}, time: {T}, valid (NDCG-10: {t_valid[0]}, HR-10: {t_valid[1]})"
                )
                print(
                    f"epoch: {epoch}, time: {T},  test (NDCG-10: {t_test[0]}, HR-10: {t_test[1]})"
                )
                if wandb_log:
                    import wandb
                    wandb.log(
                        data={
                            "val_NDCG-10": t_valid[0],
                            "val_HR-10": t_valid[1],
                            "test_NDCG-10": t_test[0],
                            "test_HR-10": t_test[1],
                        }
                    )
                # vessllogger.log(epoch, "val_NDCG-10", t_valid[0])
                # vessllogger.log(epoch, "val_HR-10", t_valid[1])
                # vessllogger.log(epoch, "test_NDCG-10", t_test[0])
                # vessllogger.log(epoch, "test_HR-10", t_test[1])

                if max_ndgc < t_test[1] and kwargs["save_path"] is not None:
                    max_ndgc = t_test[1]
                    self.save_weights(str(os.path.join(kwargs["save_path"], "best")))
                    # vessl.upload(str(os.path.join(kwargs["save_path"], "best")))

                t0.start()
        self.load_weights(str(os.path.join(kwargs["save_path"], "best")))
        return t_test

    def predict_next(self, input):
        # seq generation
        training = False
        seq = np.zeros([self.seq_max_len], dtype=np.int32)
        idx = self.seq_max_len - 1
        for i in input[::-1]:
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        input_seq = np.array([seq])
        candidate = np.expand_dims(np.arange(1, self.item_num + 1, 1), axis=0)

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
        self.save_weights(str(os.path.join(save_path, "epoch_{}".format(epoch))))
        self.load_weights(str(os.path.join(save_path, "epoch_{}".format(epoch))))


class MyRunner(vessl.RunnerBase):
    @staticmethod
    def load_model(props, artifacts):
        model_config = {
            "MAXLEN": 50,
            "NUM_BLOCKS": 2,  # NUMBER OF TRANSFORMER BLOCKS
            "HIDDEN_UNITS": 100,  # NUMBER OF UNITS IN THE ATTENTION CALCULATION
            "NUM_HEADS": 1,  # NUMBER OF ATTENTION HEADS
            "DROPOUT_RATE": 0.2,  # DROPOUT RATE
            "L2_EMB": 0.0,  # L2 REGULARIZATION COEFFICIENT
            "NUM_NEG_TEST": 100,
            # NUMBER OF NEGATIVE EXAMPLES PER POSITIVE EXAMPLE
        }

        model = SASREC_Vessl(
            item_num=12101,  # should be changed according to dataset
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

        if os.path.isfile("best.index") and os.path.isfile("best.data-00000-of-00001"):
            model.load_weights("best")

        return model

    @staticmethod
    def preprocess_data(data):
        df = np.array(list(map(int, list(pd.read_csv(BytesIO(data)).columns))))
        return df

    @staticmethod
    def predict(model, data):
        return model.predict_next(data)

    @staticmethod
    def postprocess_data(data):
        predictions = -1 * data
        rec_items = predictions.argsort()[:5]

        dic_result = {
            "Rank": [i for i in range(1, 6)],
            "ItemID": list(rec_items + 1),
            "Similarity Score": -1 * predictions[rec_items],
        }
        result = pd.DataFrame(dic_result)

        print(
            tabulate(
                result,
                headers="keys",
                tablefmt="psql",
                showindex=False,
                numalign="left",
            )
        )
        print(" ")

        time.sleep(0.5)

        output_msg = "item {}".format(rec_items[0] + 1)
        return output_msg


if __name__ == "__main__":
    vessl.configure()

    model_repository_name = "YOUR_MODEL_REPOSITORY_NAME"

    vessl.create_model_repository(name=model_repository_name)

    model_repository = vessl.read_model_repository(
        repository_name=model_repository_name,
    )

    vessl.register_model(
        repository_name=model_repository.name,
        runner_cls=MyRunner,
        requirements=["recommenders", "vessl", "tabulate"],
    )
