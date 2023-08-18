import os
import time

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from recommenders.models.sasrec.model import SASREC


class SASREC_Vessl(SASREC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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


def elapsed_time(fn, *args):
    start = time.time()
    output = fn(*args)
    end = time.time()

    elapsed = f"{end - start:.2f}"

    return elapsed, output


def load_model():
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
        model.load_weights("best").expect_partial()

    return model


def postprocess_data(data):
    predictions = -1 * data
    rec_items = predictions.argsort()[:5]

    dic_result = {
        "Rank": [i for i in range(1, 6)],
        "ItemID": list(rec_items + 1),
        "Similarity Score": -1 * predictions[rec_items],
    }
    result = pd.DataFrame(dic_result)

    time.sleep(0.5)

    best_item = rec_items[0] + 1

    return result, best_item


def main():
    st.title("Self-Attentive Sequential Recommendation(SASRec)")
    elapsed, model = elapsed_time(load_model)
    st.write(f"Model is loaded in {elapsed} seconds!")

    numbers = st.text_input(
        label="Please write input items separated by comma. (e.g. 80, 70, 100, 1)"
    )
    if numbers:
        integer_numbers = np.array(list(map(int, numbers.split(","))))
        result = model.predict_next(integer_numbers)
        table, best_item = postprocess_data(result)
        st.table(table)
        st.write(f"Best item is {best_item}")


if __name__ == "__main__":
    main()
