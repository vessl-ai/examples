import copy
import itertools
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim
import torch.utils.data
import tqdm
import vessl
from infer import infer_on_target


def _find_tensors(obj):  # pragma: no cover
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


class BaseTrainer:
    def __init__(
        self,
        exp_name="",
        gradient_clip_val=0,
        max_epoch=10000,
        log_interval=50,
    ):
        self.gradient_clip_val = gradient_clip_val
        self.on_gpu = True if torch.cuda.is_available() else False

        # training bookeeping
        self.total_batch_idx = 0
        self.running_loss = []
        self.avg_loss = 0
        self.batch_idx = 0
        self.num_val_batches = 0
        self.num_training_batches = 0
        self.num_test_batches = 0
        self.get_train_dataloader = None
        self.get_test_dataloaders = None
        self.get_val_dataloaders = None

        # training state
        self.model = None
        self.optimizers = None
        self.total_batches = 0
        self.max_epoch = max_epoch
        self.log_interval = log_interval

        self.current_epoch = 0
        self.current_it = 0
        self.exp_name = exp_name

    def fit(self, model):
        from torch.nn.parallel import DistributedDataParallel as DDP

        model.model = model.build_model()
        self.optimizers, self.lr_schedulers = model.configure_optimizers()

        device_ids = [dist.get_rank()]
        model.cuda(dist.get_rank())
        model = DDP(model, device_ids=device_ids, find_unused_parameters=True)
        self.run_pretrain_routine(model)
        return 1

    def run_pretrain_routine(self, model):
        """Sanity check a few things before starting actual training.

        :param model:
        """
        ref_model = model.module

        # give model convenience properties
        ref_model.trainer = self

        # transfer data loaders from model
        self.get_dataloaders(ref_model)

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.model = model

        # clear cache before training
        if self.on_gpu:
            torch.cuda.empty_cache()

        # CORE TRAINING LOOP
        self.train()

    def transfer_batch_to_gpu(self, batch, gpu_id):
        # base case: object can be directly moved using `cuda` or `to`
        if callable(getattr(batch, "cuda", None)):
            return batch.cuda(gpu_id, non_blocking=True)

        elif callable(getattr(batch, "to", None)):
            return batch.to(torch.device("cuda", gpu_id), non_blocking=True)

        # when list
        elif isinstance(batch, list):
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return batch

        # when tuple
        elif isinstance(batch, tuple):
            batch = list(batch)
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return tuple(batch)

        # when dict
        elif isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.transfer_batch_to_gpu(v, gpu_id)

            return batch

        # nothing matches, return the value as is without transform
        return batch

    def clip_gradients(self):
        if self.gradient_clip_val > 0:
            model = self.model
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)

    def get_dataloaders(self, model):
        self.init_train_dataloader(model)
        self.init_val_dataloader(model)

    def init_train_dataloader(self, model):
        self.get_train_dataloader = model.train_dataloader()
        self.num_training_batches = len(self.get_train_dataloader)

    def init_val_dataloader(self, model):
        self.get_val_dataloaders = model.val_dataloader()
        self.num_val_batches = len(self.get_val_dataloaders)

    def init_test_dataloader(self, model):
        self.get_val_dataloaders = model.test_dataloader()
        self.num_val_batches = len(self.get_test_dataloaders)

    def evaluate(self, model, dataloader, max_batches):
        """Run evaluation code.

        :param model: PT model
        :param dataloaders: list of PT dataloaders
        :param max_batches: Scalar
        :param test: boolean
        :return:
        """
        # enable eval mode
        model.zero_grad()
        model.eval()
        val_epoch_loss = 0

        # disable gradients to save memory
        torch.set_grad_enabled(False)

        # run training
        dl_outputs = []
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
            output = self.evaluation_forward(model, batch, batch_idx)

            # track outputs for collation
            dl_outputs.append(output)

            val_epoch_loss += output["total_loss"]

        # enable train mode again
        model.train()
        if dist.get_rank() == 0:
            vessl.log(
                step=self.current_epoch,
                payload={"valid-epoch-loss": val_epoch_loss / len(dataloader)},
            )

        # enable gradients to save memory
        torch.set_grad_enabled(True)

        return dl_outputs

    def run_evaluation(self):
        # when testing make sure user defined a test step
        model = self.model

        # val
        dataloader = self.get_val_dataloaders
        max_batches = self.num_val_batches

        # run evaluation
        eval_results = self.evaluate(self.model, dataloader, max_batches)

    def evaluation_forward(self, model, batch, batch_idx):
        # make dataloader_idx arg in validation_step optional

        output = model.module.validation_step(batch)

        return output

    def train(self):
        model = self.model
        dataset_path = "/input/vessl-diff-svc"
        output_path = f"/output/{self.exp_name}"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        # run all epochs
        for epoch in range(self.current_epoch, self.max_epoch):
            self.current_epoch = epoch

            # update training progress in trainer and model
            model.module.current_epoch = epoch

            total_val_batches = 0

            # total batches includes multiple val checks
            self.total_batches = self.num_training_batches

            self.run_training_epoch()

            # update LR schedulers
            if self.lr_schedulers is not None:
                for lr_scheduler in self.lr_schedulers:
                    lr_scheduler.step(epoch=self.current_epoch)
                    if dist.get_rank() == 0:
                        vessl.log(
                            step=self.current_epoch,
                            payload={"learning-rate": lr_scheduler.get_lr()[0]},
                        )

            if (epoch + 1) % self.log_interval == 0 and dist.get_rank() == 0:
                print("===== Running evaluation =====")
                self.run_evaluation()
                torch.save(model.state_dict(), f"{output_path}/epoch_{epoch+1}.pt")
                # vessl.log(payload={f"epoch-{epoch+1}": model.state_dict()})

                logging_dir = f"{dataset_path}/assets/logging"

                for file in os.listdir(logging_dir):
                    audio_path = f"{logging_dir}/{file}"
                    save_path = f"{output_path}/infer_{file}_epoch_{epoch+1}.wav"
                    print(f"===== Inference for {file} ===== ")

                    model_path = f"{output_path}/epoch_{epoch+1}.pt"
                    infer_on_target(self.exp_name, model_path, audio_path, save_path)
                    vessl.log(
                        payload={
                            f"audio": [
                                vessl.Audio(
                                    save_path, caption=f"infer {file} epoch {epoch+1}"
                                )
                            ]
                        }
                    )

    def run_training_epoch(self):
        epoch_loss = 0

        # before epoch hook
        model = self.model.module
        model.on_epoch_start()

        # run epoch
        for batch_idx, batch in enumerate(tqdm.tqdm(self.get_train_dataloader)):
            self.batch_idx = batch_idx

            output = self.run_training_batch(batch, batch_idx)
            epoch_loss += output["loss"].item()

            # iteration-wise loss
            if dist.get_rank() == 0:
                vessl.log(
                    step=self.current_it,
                    payload={"train-iteration-loss": output["loss"].item()},
                )

            self.total_batch_idx += 1
            self.current_it += 1
            model.global_step = self.current_it
        # epoch end hook
        if dist.get_rank() == 0:
            model.on_epoch_end()
            vessl.log(
                step=self.current_epoch,
                payload={
                    "train-epoch-loss": epoch_loss / len(self.get_train_dataloader)
                },
            )

    def run_training_batch(self, batch, batch_idx):
        self.hiddens = None

        optimizer = self.optimizers[0]
        output = self.training_forward(batch, batch_idx)
        self.model.module.backward(output["loss"])
        self.model.module.optimizer_step(self.current_epoch, batch_idx, optimizer)

        return output

    def training_forward(self, batch, batch_idx):
        """
        Handle forward for each training case (distributed, single gpu, etc...)
        :param batch:
        :param batch_idx:
        :return:
        """

        # enable not needing to add opt_idx to training_step
        args = [batch, batch_idx]

        gpu_id = dist.get_rank()
        batch = self.transfer_batch_to_gpu(copy.copy(batch), gpu_id)
        output = self.model.module.training_step(*args)

        return output
