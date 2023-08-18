import matplotlib

matplotlib.use("Agg")

import logging
import os
import random
import sys

import numpy as np
import torch.utils.data
import utils
from torch import nn
from utils.hparams import hparams, set_hparams
from utils.pl_utils import BaseTrainer

torch.multiprocessing.set_sharing_strategy(
    os.getenv("TORCH_SHARE_STRATEGY", "file_system")
)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)


class BaseTask(nn.Module):
    """
    Base class for training tasks.
    1. *load_ckpt*:
        load checkpoint;
    2. *training_step*:
        record and log the loss;
    3. *optimizer_step*:
        run backwards step;
    4. *start*:
        load training configs, backup code, log to tensorboard, start training;
    5. *configure_ddp* and *init_ddp_connection*:
        start parallel training.

    Subclasses should define:
    1. *build_model*, *build_optimizer*, *build_scheduler*:
        how to build the model, the optimizer and the training scheduler;
    2. *_training_step*:
        one training step of the model;
    3. *validation_end* and *_validation_end*:
        postprocess the validation output.
    """

    def __init__(self, *args, **kwargs):
        # dataset configs
        super(BaseTask, self).__init__(*args, **kwargs)
        self.current_epoch = 0
        self.global_step = 0
        self.loaded_optimizer_states_dict = {}
        self.trainer = None
        self.logger = None
        self.on_gpu = False
        self.use_dp = False
        self.use_ddp = False
        self.example_input_array = None

        self.max_tokens = hparams["max_tokens"]
        self.max_sentences = hparams["max_sentences"]
        self.max_eval_tokens = hparams["max_eval_tokens"]
        if self.max_eval_tokens == -1:
            hparams["max_eval_tokens"] = self.max_eval_tokens = self.max_tokens
        self.max_eval_sentences = hparams["max_eval_sentences"]
        if self.max_eval_sentences == -1:
            hparams["max_eval_sentences"] = self.max_eval_sentences = self.max_sentences

        self.model = None
        self.training_losses_meter = None

    ###########
    # Training, validation and testing
    ###########
    def build_model(self):
        raise NotImplementedError

    def load_ckpt(
        self,
        ckpt_base_dir,
        current_model_name=None,
        model_name="model",
        force=True,
        strict=True,
    ):
        # This function is updated on 2021.12.13
        if current_model_name is None:
            current_model_name = model_name
        utils.load_ckpt(
            self.__getattr__(current_model_name),
            ckpt_base_dir,
            current_model_name,
            force,
            strict,
        )

    def on_epoch_start(self):
        self.training_losses_meter = {"total_loss": utils.AvgrageMeter()}

    def _training_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: total loss: torch.Tensor, loss_log: dict
        """
        raise NotImplementedError

    def training_step(self, sample, batch_idx):
        loss_ret = self._training_step(sample, batch_idx)
        if loss_ret is None:
            return {"loss": None}
        total_loss, log_outputs = loss_ret
        log_outputs = utils.tensors_to_scalars(log_outputs)
        for k, v in log_outputs.items():
            if k not in self.training_losses_meter:
                self.training_losses_meter[k] = utils.AvgrageMeter()
            if not np.isnan(v):
                self.training_losses_meter[k].update(v)
        self.training_losses_meter["total_loss"].update(total_loss.item())

        try:
            log_outputs["lr"] = self.scheduler.get_lr()
            if isinstance(log_outputs["lr"], list):
                log_outputs["lr"] = log_outputs["lr"][0]
        except:
            pass

        return {
            "loss": total_loss,
        }

    def optimizer_step(self, epoch, batch_idx, optimizer):
        optimizer.step()
        optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step(self.global_step)

    def on_epoch_end(self):
        loss_outputs = {
            k: round(v.avg, 4) for k, v in self.training_losses_meter.items()
        }
        print(
            f"\n==============\n "
            f"Epoch {self.current_epoch} ended. Steps: {self.global_step}. {loss_outputs}"
            f"\n==============\n"
        )

    def validation_step(self, sample):
        """

        :param sample:
        :param batch_idx:
        :return: output: dict
        """
        raise NotImplementedError

    def _validation_end(self, outputs):
        """

        :param outputs:
        :return: loss_output: dict
        """
        raise NotImplementedError

    def validation_end(self, outputs):
        loss_output = self._validation_end(outputs)
        print(
            f"\n==============\n " f"valid results: {loss_output}" f"\n==============\n"
        )
        return {
            "log": {f"val/{k}": v for k, v in loss_output.items()},
            "val_loss": loss_output["total_loss"],
        }

    def build_scheduler(self, optimizer):
        raise NotImplementedError

    def build_optimizer(self, model):
        raise NotImplementedError

    def configure_optimizers(self):
        optm = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(optm)
        return [optm], [self.scheduler]

    ###########
    # Running configuration
    ###########

    @classmethod
    def start(cls):
        set_hparams()
        os.environ["MASTER_PORT"] = str(random.randint(15000, 30000))
        random.seed(hparams["seed"])
        np.random.seed(hparams["seed"])
        task = cls()
        set_hparams()
        trainer = BaseTrainer(
            exp_name=hparams["exp_name"],
            gradient_clip_val=hparams["clip_grad_norm"],
            max_epoch=hparams["max_epoch"],
            log_interval=hparams["log_interval"],
        )
        trainer.fit(task)

    def train_dataloader(self):
        return None

    def test_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def backward(self, loss):
        loss.backward()
