import argparse
import os

import vessl
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.constants import (
    SEED,
    DEFAULT_PREDICTION_COL as PREDICT_COL
)
from recommenders.datasets.python_splitters import python_random_split
from recommenders.datasets.pandas_df_utils import user_item_pairs
from recommenders.utils import tf_utils, gpu_utils, plot
import recommenders.evaluation.python_evaluation as evaluator
import recommenders.models.wide_deep.wide_deep_utils as wide_deep

vessl.init()


# Columns
USER_COL = 'userId'
ITEM_COL = 'movieId'
RATING_COL = 'rating'
ITEM_FEAT_COL = 'genres'

# Recommend top k items
TOP_K = 10

# Metrics to use for evaluation
RANKING_METRICS = [
    evaluator.ndcg_at_k.__name__,
    evaluator.precision_at_k.__name__,
]
RATING_METRICS = [
    evaluator.rmse.__name__,
    evaluator.mae.__name__,
]

# Set seed for deterministic result
RANDOM_SEED = SEED

# Use session hook to evaluate model while training
EVALUATE_WHILE_TRAINING = True

#### Hyperparameters
MODEL_TYPE = 'wide_deep'
STEPS = 50000  # Number of batches to train
BATCH_SIZE = 32

# Wide (linear) model hyperparameters
LINEAR_OPTIMIZER = 'adagrad'
LINEAR_OPTIMIZER_LR = 0.0621  # Learning rate
LINEAR_L1_REG = 0.0           # Regularization rate for FtrlOptimizer
LINEAR_L2_REG = 0.0
LINEAR_MOMENTUM = 0.0         # Momentum for MomentumOptimizer or RMSPropOptimizer

# DNN model hyperparameters
DNN_OPTIMIZER = 'adadelta'
DNN_OPTIMIZER_LR = 0.1
DNN_L1_REG = 0.0           # Regularization rate for FtrlOptimizer
DNN_L2_REG = 0.0
DNN_MOMENTUM = 0.0         # Momentum for MomentumOptimizer or RMSPropOptimizer

# Layer dimensions. Defined as follows to make this notebook runnable from Hyperparameter tuning services like AzureML Hyperdrive
DNN_HIDDEN_LAYER_1 = 0     # Set 0 to not use this layer
DNN_HIDDEN_LAYER_2 = 64    # Set 0 to not use this layer
DNN_HIDDEN_LAYER_3 = 128   # Set 0 to not use this layer
DNN_HIDDEN_LAYER_4 = 512   # Note, at least one layer should have nodes.
DNN_HIDDEN_UNITS = [h for h in [DNN_HIDDEN_LAYER_1, DNN_HIDDEN_LAYER_2, DNN_HIDDEN_LAYER_3, DNN_HIDDEN_LAYER_4] if h > 0]
DNN_USER_DIM = 32          # User embedding feature dimension
DNN_ITEM_DIM = 16          # Item embedding feature dimension
DNN_DROPOUT = 0.8
DNN_BATCH_NORM = 1         # 1 to use batch normalization, 0 if not.


class VesslLogger:
    """VESSL logger"""

    def __init__(self):
        """Initializer"""
        self._log = {}

    def log(self, metric, value):
        """Log metrics. Each metric's log will be stored in the corresponding list.
        Args:
            metric (str): Metric name.
            value (float): Value.
        """
        if metric not in self._log:
            self._log[metric] = []
        self._log[metric].append(value)
        vessl.log({
            metric: value,
        })

    def get_log(self):
        """Getter
        Returns:
            dict: Log metrics.
        """
        return self._log


def load_data(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    raw_data = pd.read_csv(data_path)
    return raw_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--input-path', type=str, default='/input',
                        help='input dataset path')
    parser.add_argument('--output-path', type=str, default='/output',
                        help='output files path')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='save the current model')
    args = parser.parse_args()

    print("Tensorflow Version:", tf.__version__)
    print("GPUs:\n", gpu_utils.get_gpu_info())

    # Load dataset
    data = load_data(args.input_path, "ratings.csv")
    movies = load_data(args.input_path, "movies.csv")

    # Preprocess dataset
    new_data = data.set_index(ITEM_COL).join(movies.set_index(ITEM_COL)).filter(
        [USER_COL, ITEM_COL, RATING_COL, ITEM_FEAT_COL]).reset_index()
    genres_encoder = sklearn.preprocessing.MultiLabelBinarizer()
    new_data[ITEM_FEAT_COL] = genres_encoder.fit_transform(
        new_data[ITEM_FEAT_COL].apply(lambda s: s.split("|"))
    ).tolist()
    train, test = python_random_split(new_data, ratio=0.75, seed=RANDOM_SEED)

    # Unique items in the dataset
    items = new_data.drop_duplicates(ITEM_COL)[[ITEM_COL, ITEM_FEAT_COL]].reset_index(drop=True)
    item_feat_shape = len(items[ITEM_FEAT_COL][0])
    # Unique users in the dataset
    users = new_data.drop_duplicates(USER_COL)[[USER_COL]].reset_index(drop=True)
    print("Total {} items and {} users in the dataset".format(len(items), len(users)))

    # Define wide (linear) and deep (dnn) features
    wide_columns, deep_columns = wide_deep.build_feature_columns(
        users=users[USER_COL].values,
        items=items[ITEM_COL].values,
        user_col=USER_COL,
        item_col=ITEM_COL,
        item_feat_col=ITEM_FEAT_COL,
        crossed_feat_dim=1000,
        user_dim=DNN_USER_DIM,
        item_dim=DNN_ITEM_DIM,
        item_feat_shape=item_feat_shape,
        model_type=MODEL_TYPE,
    )

    print("Wide feature specs:")
    for c in wide_columns:
        print("\t", str(c)[:100], "...")
    print("Deep feature specs:")
    for c in deep_columns:
        print("\t", str(c)[:100], "...")

    # Create model checkpoint every n steps. We store the model 5 times.
    save_checkpoints_steps = max(1, STEPS // 5)

    # Build a model based on the parameters
    model_dir = args.output_path
    model = wide_deep.build_model(
        model_dir=model_dir,
        wide_columns=wide_columns,
        deep_columns=deep_columns,
        linear_optimizer=tf_utils.build_optimizer(LINEAR_OPTIMIZER, LINEAR_OPTIMIZER_LR, **{
            'l1_regularization_strength': LINEAR_L1_REG,
            'l2_regularization_strength': LINEAR_L2_REG,
            'momentum': LINEAR_MOMENTUM,
        }),
        dnn_optimizer=tf_utils.build_optimizer(DNN_OPTIMIZER, DNN_OPTIMIZER_LR, **{
            'l1_regularization_strength': DNN_L1_REG,
            'l2_regularization_strength': DNN_L2_REG,
            'momentum': DNN_MOMENTUM,
        }),
        dnn_hidden_units=DNN_HIDDEN_UNITS,
        dnn_dropout=DNN_DROPOUT,
        dnn_batch_norm=(DNN_BATCH_NORM == 1),
        log_every_n_iter=STEPS // 1000,
        save_checkpoints_steps=STEPS // 1000,
        seed=RANDOM_SEED
    )

    # Prepare ranking evaluation set, i.e. get the cross join of all user-item pairs
    ranking_pool = user_item_pairs(
        user_df=users,
        item_df=items,
        user_col=USER_COL,
        item_col=ITEM_COL,
        user_item_filter_df=train,  # Remove seen items
        shuffle=True,
        seed=RANDOM_SEED
    )

    # Define training hooks to track performance while training
    hooks = []
    cols = {
        'col_user': USER_COL,
        'col_item': ITEM_COL,
        'col_rating': RATING_COL,
        'col_prediction': PREDICT_COL,
    }

    if EVALUATE_WHILE_TRAINING:
        evaluation_logger = VesslLogger()
        for metrics in (RANKING_METRICS, RATING_METRICS):
            if len(metrics) > 0:
                hooks.append(
                    tf_utils.evaluation_log_hook(
                        model,
                        logger=evaluation_logger,
                        true_df=test,
                        y_col=RATING_COL,
                        eval_df=ranking_pool if metrics == RANKING_METRICS else test.drop(RATING_COL, axis=1),
                        every_n_iter=STEPS // 1000,
                        model_dir=model_dir,
                        eval_fns=[evaluator.metrics[m] for m in metrics],
                        **({**cols, 'k': TOP_K} if metrics == RANKING_METRICS else cols)
                    )
                )

    # Define training input (sample feeding) function
    train_fn = tf_utils.pandas_input_fn(
        df=train,
        y_col=RATING_COL,
        batch_size=BATCH_SIZE,
        num_epochs=None,  # We use steps=TRAIN_STEPS instead.
        shuffle=True,
        seed=RANDOM_SEED,
    )

    print(
        "Training steps = {}, Batch size = {} (num epochs = {})"
            .format(STEPS, BATCH_SIZE, (STEPS * BATCH_SIZE) // len(train))
    )
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    try:
        model.train(
            input_fn=train_fn,
            hooks=hooks,
            steps=STEPS
        )
    except tf.train.NanLossDuringTrainingError:
        import warnings

        warnings.warn(
            "Training stopped with NanLossDuringTrainingError. "
            "Try other optimizers, smaller batch size and/or smaller learning rate."
        )

    if args.save_model:
        exported_path = tf_utils.export_model(
            model=model,
            train_input_fn=train_fn,
            eval_input_fn=tf_utils.pandas_input_fn(
                df=test, y_col=RATING_COL
            ),
            tf_feat_cols=wide_columns + deep_columns,
            base_dir=args.output_path
        )
        print("Model exported to", str(exported_path))
