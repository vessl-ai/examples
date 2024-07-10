import argparse
import gc
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PMAP_USE_TENSORSTORE"] = "false"

# from collections import deque
import jax
import numpy as np
import pandas as pd
import timesfm
import vessl
from jax import numpy as jnp
from paxml import checkpoint_types, checkpoints, learners, tasks_lib, trainer_lib
from praxis import optimizers, pax_fiddle, py_utils, schedules
from timesfm import data_loader, patched_decoder
from tqdm import tqdm

DATA_DICT = {
    "ettm2": {
        "boundaries": [34560, 46080, 57600],
        "data_path": "./datasets/ETT-small/ETTm2.csv",
        "freq": "15min",
    },
    "ettm1": {
        "boundaries": [34560, 46080, 57600],
        "data_path": "./datasets/ETT-small/ETTm1.csv",
        "freq": "15min",
    },
    "etth2": {
        "boundaries": [8640, 11520, 14400],
        "data_path": "./datasets/ETT-small/ETTh2.csv",
        "freq": "H",
    },
    "etth1": {
        "boundaries": [8640, 11520, 14400],
        "data_path": "./datasets/ETT-small/ETTh1.csv",
        "freq": "H",
    },
    "elec": {
        "boundaries": [18413, 21044, 26304],
        "data_path": "./datasets/electricity/electricity.csv",
        "freq": "H",
    },
    "traffic": {
        "boundaries": [12280, 14036, 17544],
        "data_path": "./datasets/traffic/traffic.csv",
        "freq": "H",
    },
    "weather": {
        "boundaries": [36887, 42157, 52696],
        "data_path": "./datasets/weather/weather.csv",
        "freq": "10min",
    },
}


def load_timesfm_model(context_length: int = 512) -> timesfm.TimesFm:
    timesfm_model = timesfm.TimesFm(
        context_len=context_length,
        horizon_len=128,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="gpu",
    )
    timesfm_model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

    return timesfm_model


def load_dataset(
    dataset_name: str, context_length: int = 512, pred_length: int = 96
) -> data_loader.TimeSeriesdata:
    assert dataset_name in DATA_DICT

    data_path = DATA_DICT[dataset_name]["data_path"]
    freq = DATA_DICT[dataset_name]["freq"]
    # int_freq = timesfm.freq_map(freq)
    boundaries = DATA_DICT[dataset_name]["boundaries"]

    data_df = pd.read_csv(open(data_path, "r"))

    ts_cols = [col for col in data_df.columns if col != "date"]
    num_cov_cols = None
    cat_cov_cols = None

    num_ts = len(ts_cols)

    ts_data = data_loader.TimeSeriesdata(
        data_path=data_path,
        datetime_col="date",
        num_cov_cols=num_cov_cols,
        cat_cov_cols=cat_cov_cols,
        ts_cols=np.array(ts_cols),
        train_range=[0, boundaries[0]],
        val_range=[boundaries[0], boundaries[1]],
        test_range=[boundaries[1], boundaries[2]],
        hist_len=context_length,
        pred_len=pred_length,
        batch_size=num_ts,
        freq=freq,
        normalize=True,
        epoch_len=None,
        holiday=False,
        permute=True,
    )

    return ts_data


def process_train_batch(batch, batch_size, num_ts):
    past_ts = batch[0].reshape(batch_size * num_ts, -1)
    actual_ts = batch[3].reshape(batch_size * num_ts, -1)
    return py_utils.NestedMap(input_ts=past_ts, actual_ts=actual_ts)


def process_eval_batch(batch):
    past_ts = batch[0]
    actual_ts = batch[3]
    return py_utils.NestedMap(input_ts=past_ts, actual_ts=actual_ts)


@pax_fiddle.auto_config
def build_learner() -> learners.Learner:
    return pax_fiddle.Config(
        learners.Learner,
        name="learner",
        loss_name="avg_qloss",
        optimizer=optimizers.Adam(
            epsilon=1e-7,
            clip_threshold=1e2,
            learning_rate=1e-2,
            lr_schedule=pax_fiddle.Config(
                schedules.Cosine,
                initial_value=1e-3,
                final_value=1e-4,
                total_steps=40000,
            ),
            ema_decay=0.9999,
        ),
        # Linear probing i.e we hold the transformer layers fixed.
        bprop_variable_exclusion=[".*/stacked_transformer_layer/.*"],
    )


def get_last_batch(batches):
    for batch in batches.as_numpy_iterator():
        pass
    # use this for the large iterator
    # batch = deque(batches.as_numpy_iterator(), 1)

    return batch


def init_model_state(timesfm_model, task, train_batches, key, batch_size, num_ts):
    last_batch = get_last_batch(train_batches)
    key, init_key = jax.random.split(key)
    jax_model_states, _ = trainer_lib.initialize_model_state(
        task,
        init_key,
        process_train_batch(last_batch, batch_size, num_ts),
        checkpoint_type=checkpoint_types.CheckpointType.GDA,
    )
    jax_model_states.mdl_vars["params"]["core_layer"] = (
        timesfm_model._train_state.mdl_vars["params"]
    )
    # jax_vars = jax_model_states.mdl_vars
    gc.collect()

    return jax_model_states, key


def get_steps_and_seeds(task, key):
    def train_step(states, prng_key, inputs):
        return trainer_lib.train_step_single_learner(task, states, prng_key, inputs)

    def eval_step(states, prng_key, inputs):
        states = states.to_eval_state()
        return trainer_lib.eval_step_single_learner(task, states, prng_key, inputs)

    key, train_key, eval_key = jax.random.split(key, 3)
    train_prng_seed = jax.random.split(train_key, num=jax.local_device_count())
    eval_prng_seed = jax.random.split(eval_key, num=jax.local_device_count())

    p_train_step = jax.pmap(train_step, axis_name="batch")
    p_eval_step = jax.pmap(eval_step, axis_name="batch")

    return p_train_step, p_eval_step, train_prng_seed, eval_prng_seed


def reshape_batch_for_pmap(batch, num_devices):
    def _reshape(input_tensor):
        bsize = input_tensor.shape[0]
        residual_shape = list(input_tensor.shape[1:])
        nbsize = bsize // num_devices
        return jnp.reshape(input_tensor, [num_devices, nbsize] + residual_shape)

    return jax.tree.map(_reshape, batch)


def compute_mean_mae(timesfm_model, batches):
    mae_losses = []
    for batch in tqdm(batches.as_numpy_iterator()):
        past = batch[0]
        actuals = batch[3]
        _, forecasts = timesfm_model.forecast(list(past), [0] * past.shape[0])
        forecasts = forecasts[:, 0 : actuals.shape[1], 5]
        mae_losses.append(np.abs(forecasts - actuals).mean())

    return np.mean(mae_losses)


def train(
    context_length: int = 512,
    pred_length: int = 96,
    batch_size: int = 16,
    num_epochs: int = 100,
    patience_threshold: int = 5,
    eval_steps: int = 1000,
    checkpoint_dir: str = "./checkpoints",
):
    # load pretrained model
    timesfm_model = load_timesfm_model(context_length)

    # prepare learner
    model = pax_fiddle.Config(
        patched_decoder.PatchedDecoderFinetuneModel,
        name="patched_decoder_finetune",
        core_layer_tpl=timesfm_model.model_p,
    )
    task_p = tasks_lib.SingleTask(
        name="ts-learn",
        model=model,
        train=tasks_lib.SingleTask.Train(
            learner=build_learner(),
        ),
    )
    task_p.model.ici_mesh_shape = [1, 1, 1]
    task_p.model.mesh_axis_names = ["replica", "data", "mdl"]
    num_devices = jax.local_device_count()

    # load dataset
    dataset = load_dataset(
        "ettm1", context_length=context_length, pred_length=pred_length
    )
    train_batches = dataset.tf_dataset(mode="train", shift=1).batch(batch_size)
    val_batches = dataset.tf_dataset(mode="val", shift=pred_length)
    test_batches = dataset.tf_dataset(mode="test", shift=pred_length)
    num_ts = dataset.batch_size

    mean_mae_before_ft = compute_mean_mae(timesfm_model, test_batches)
    print(f"Mean MAE before finetuning: {mean_mae_before_ft}")

    # initialize model
    initial_key = jax.random.PRNGKey(seed=1234)
    jax_model_states, key = init_model_state(
        timesfm_model, task_p, train_batches, initial_key, batch_size, num_ts
    )

    # training loop
    p_train_step, p_eval_step, train_prng_seed, eval_prng_seed = get_steps_and_seeds(
        task_p, key
    )
    replicated_jax_states = trainer_lib.replicate_model_state(jax_model_states)
    # replicated_jax_vars = replicated_jax_states.mdl_vars

    best_eval_loss = 1e7
    patience = 0
    step_count = 0
    for epoch in range(num_epochs):
        print(f"__________________Epoch: {epoch}__________________", flush=True)
        train_its = train_batches.as_numpy_iterator()
        if patience >= patience_threshold:
            print("Early stopping.", flush=True)
            break
        for batch in tqdm(train_its):
            train_losses = []
            if patience >= patience_threshold:
                print("Early stopping.", flush=True)
                break
            tbatch = process_train_batch(batch, batch_size, num_ts)
            tbatch = reshape_batch_for_pmap(tbatch, num_devices)
            replicated_jax_states, step_fun_out = p_train_step(
                replicated_jax_states, train_prng_seed, tbatch
            )
            train_losses.append(step_fun_out.loss[0])
            if step_count % eval_steps == 0:
                vessl.log(
                    payload={"train_loss": np.mean(train_losses)}, step=step_count
                )
                print(
                    f"Train loss at step {step_count}: {np.mean(train_losses)}",
                    flush=True,
                )
                train_losses = []
                print("Starting eval.", flush=True)
                val_its = val_batches.as_numpy_iterator()
                eval_losses = []
                for ev_batch in tqdm(val_its):
                    ebatch = process_eval_batch(ev_batch)
                    ebatch = reshape_batch_for_pmap(ebatch, num_devices)
                    _, step_fun_out = p_eval_step(
                        replicated_jax_states, eval_prng_seed, ebatch
                    )
                    eval_losses.append(step_fun_out.loss[0])
                mean_loss = np.mean(eval_losses)
                vessl.log(payload={"eval_loss": mean_loss}, step=step_count)
                print(f"Eval loss at step {step_count}: {mean_loss}", flush=True)
                if mean_loss < best_eval_loss or np.isnan(mean_loss):
                    best_eval_loss = mean_loss
                    print("Saving checkpoint.")
                    jax_state_for_saving = (
                        py_utils.maybe_unreplicate_for_fully_replicated(
                            replicated_jax_states
                        )
                    )
                    checkpoints.save_checkpoint(
                        jax_state_for_saving, checkpoint_dir, overwrite=True
                    )
                    patience = 0
                    del jax_state_for_saving
                    gc.collect()
                else:
                    patience += 1
                    print(f"patience: {patience}")
            step_count += 1

    train_state = checkpoints.restore_checkpoint(jax_model_states, checkpoint_dir)
    print(train_state.step)
    timesfm_model._train_state.mdl_vars["params"] = train_state.mdl_vars["params"][
        "core_layer"
    ]
    timesfm_model.jit_decode()

    mean_mae_after_ft = compute_mean_mae(timesfm_model, test_batches)
    print(f"Mean MAE after finetuning: {mean_mae_after_ft}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--pred-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--patience-threshold", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    train(
        context_length=args.context_length,
        pred_length=args.pred_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        patience_threshold=args.patience_threshold,
        eval_steps=args.eval_steps,
        checkpoint_dir=args.checkpoint_dir,
    )
