import argparse
import os
import time
from io import BytesIO

import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI, UploadFile
from tabulate import tabulate
from model import SASREC_Vessl

app = FastAPI()


def load_model(model_path):
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

    index_path = os.path.join(model_path, "best.index")
    data_path = os.path.join(model_path, "best.data-00000-of-00001")
    if os.path.isfile(index_path) and os.path.isfile(data_path):
        model.load_weights(str(os.path.join(model_path,"best")))

    return model


@app.post("/predict/")
async def predict(file: UploadFile):
    csv_data = await file.read()

    # preprocess csv data
    df = np.array(list(map(int, list(pd.read_csv(BytesIO(csv_data)).columns))))

    # inference
    res = model.predict_next(df)

    # postprocess result
    predictions = -1 * res
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path')
    args = parser.parse_args()

    model = load_model(args.model_path)

    uvicorn.run(app, host="0.0.0.0", port=5000)
