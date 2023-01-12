import os
import numpy as np
import tensorflow as tf
import vessl

from recommenders.utils.timer import Timer
from recommenders.models.sasrec.model import SASREC
from recommenders.models.sasrec.ssept import SSEPT

from tqdm import tqdm


class VesslLogger:
    """VESSL logger"""
    def __init__(self):
        """Initializer"""
        self._log = {}

    def log(self, step , metric, value):
        """Log metrics. Each metric's log will be stored in the corresponding list.
        Args:
            metric (str): Metric name.
            value (float): Value.
        """
        if metric not in self._log:
            self._log[metric] = []
        self._log[metric].append(value)
        vessl.log(step=step, payload={
            metric: value,
        })

    def get_log(self):
        """Getter
        Returns:
            dict: Log metrics.
        """
        return self._log


class SASREC_Vessl(SASREC)  :

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
                pos_logits, neg_logits, loss_mask = self
                loss = loss_function(pos_logits, neg_logits, loss_mask)

            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            # vessl.hp.lr = optimizer.learning_rate
            # vessl.hp.update()

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
        self.load_weights(str(os.path.join(kwargs['save_path'] , 'best')))
        return t_test

    def predict_next(self, input):
        # seq generation
        training = False
        seq = np.zeros([self.seq_max_len], dtype=np.int32)
        idx = self.seq_max_len - 1
        idx -= 1
        for i in reversed(input):
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
        vessl.upload(str(os.path.join(save_path, 'epoch_{}.data-00000-of-00001'.format(epoch))))
        vessl.upload(str(os.path.join(save_path, 'epoch_{}.index'.format(epoch))))
