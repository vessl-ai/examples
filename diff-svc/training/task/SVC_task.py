import os

import numpy as np
import torch
import torch.distributed as dist
import utils
from network.diff.candidate_decoder import FFT
from network.diff.diffusion import GaussianDiffusion
from network.diff.net import DiffNet
from network.vocoders.base_vocoder import BaseVocoder, get_vocoder_cls
from preprocessing.hubertinfer import Hubertencoder
from training.dataset.base_dataset import BaseDataset
from training.task.base_task import BaseTask
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset
from utils.pitch_utils import denorm_f0, norm_interp_f0

DIFF_DECODERS = {
    "wavenet": lambda hp: DiffNet(hp["audio_num_mel_bins"]),
    "fft": lambda hp: FFT(
        hp["hidden_size"], hp["dec_layers"], hp["dec_ffn_kernel_size"], hp["num_heads"]
    ),
}


class SVCDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        exp_name = hparams["exp_name"]
        self.data_dir = f"./raw/{exp_name}/binary"
        self.prefix = prefix
        self.hparams = hparams
        self.sizes = np.load(f"{self.data_dir}/{self.prefix}_lengths.npy")
        self.indexed_ds = None

        # pitch stats
        f0_stats_fn = f"{self.data_dir}/train_f0s_mean_std.npy"
        if os.path.exists(f0_stats_fn):
            hparams["f0_mean"], hparams["f0_std"] = self.f0_mean, self.f0_std = np.load(
                f0_stats_fn
            )
            hparams["f0_mean"] = float(hparams["f0_mean"])
            hparams["f0_std"] = float(hparams["f0_std"])
        else:
            hparams["f0_mean"], hparams["f0_std"] = self.f0_mean, self.f0_std = (
                None,
                None,
            )

    def _get_item(self, index):
        if hasattr(self, "avail_idxs") and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        max_frames = hparams["max_frames"]
        spec = torch.Tensor(item["mel"])[:max_frames]
        energy = (spec.exp() ** 2).sum(-1).sqrt()
        mel2ph = (
            torch.LongTensor(item["mel2ph"])[:max_frames] if "mel2ph" in item else None
        )
        f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
        hubert = torch.Tensor(item["hubert"][: hparams["max_input_tokens"]])
        pitch = torch.LongTensor(item.get("pitch"))[:max_frames]

        sample = {
            "id": index,
            "item_name": item["item_name"],
            "hubert": hubert,
            "mel": spec,
            "pitch": pitch,
            "energy": energy,
            "f0": f0,
            "uv": uv,
            "mel2ph": mel2ph,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }

        return sample

    def collater(self, samples):
        from preprocessing.process_pipeline import File2Batch

        return File2Batch.processed_input2batch(samples)


class SVCTask(BaseTask):
    def __init__(self):
        super(SVCTask, self).__init__()
        self.dataset_cls = SVCDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
        self.phone_encoder = Hubertencoder(hparams["hubert_path"])

    def build_dataloader(
        self,
        dataset,
        shuffle,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=-1,
        endless=False,
        batch_by_size=True,
    ):
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if max_tokens is not None:
            max_tokens *= devices_cnt
        if max_sentences is not None:
            max_sentences *= devices_cnt
        indices = dataset.ordered_indices()
        if batch_by_size:
            batch_sampler = utils.batch_by_size(
                indices,
                dataset.num_tokens,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler = []
            for i in range(0, len(indices), max_sentences):
                batch_sampler.append(indices[i : i + max_sentences])

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [
                    b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))
                ]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers

        # set DDP as default option

        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
        batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collater,
            batch_sampler=batches,
            num_workers=num_workers,
            pin_memory=False,
        )

    def train_dataloader(self):
        train_dataset = self.dataset_cls(hparams["train_set_name"], shuffle=True)
        return self.build_dataloader(
            train_dataset,
            True,
            self.max_tokens,
            self.max_sentences,
            endless=hparams["endless_ds"],
        )

    def val_dataloader(self):
        valid_dataset = self.dataset_cls(hparams["valid_set_name"], shuffle=False)
        return self.build_dataloader(
            valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences
        )

    def test_dataloader(self):
        test_dataset = self.dataset_cls(hparams["test_set_name"], shuffle=False)
        return self.build_dataloader(
            test_dataset,
            False,
            self.max_eval_tokens,
            self.max_eval_sentences,
            batch_by_size=False,
        )

    def build_tts_model(self):
        mel_bins = hparams["audio_num_mel_bins"]
        self.model = GaussianDiffusion(
            phone_encoder=self.phone_encoder,
            out_dims=mel_bins,
            denoise_fn=DIFF_DECODERS[hparams["diff_decoder_type"]](hparams),
            timesteps=hparams["timesteps"],
            K_step=hparams["K_step"],
            loss_type=hparams["diff_loss_type"],
            spec_min=hparams["spec_min"],
            spec_max=hparams["spec_max"],
        )

    def build_model(self):
        self.build_tts_model()
        if hparams["load_ckpt"] != "":
            self.load_ckpt(hparams["load_ckpt"], strict=True)
        utils.print_arch(self.model)
        return self.model

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hparams["lr"],
            betas=(hparams["optimizer_adam_beta1"], hparams["optimizer_adam_beta2"]),
            weight_decay=hparams["weight_decay"],
        )
        return optimizer

    def _validation_end(self, outputs):
        all_losses_meter = {
            "total_loss": utils.AvgrageMeter(),
        }
        for output in outputs:
            n = output["nsamples"]
            for k, v in output["losses"].items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = utils.AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter["total_loss"].update(output["total_loss"], n)
        return {k: round(v.avg, 4) for k, v in all_losses_meter.items()}

    def run_model(self, model, sample, return_output=False, infer=False):
        """
        steps:
        1. run the full model, calc the main loss
        2. calculate loss for dur_predictor, pitch_predictor, energy_predictor
        """
        hubert = sample["hubert"].cuda()  # [B, T_t,H]
        target = sample["mels"].cuda()  # [B, T_s, 80]
        mel2ph = sample["mel2ph"].cuda()  # [B, T_s]
        f0 = sample["f0"].cuda()
        uv = sample["uv"].cuda()
        energy = sample["energy"].cuda()
        spk_embed = sample.get("spk_embed")

        output = model(
            hubert,
            mel2ph=mel2ph,
            spk_embed=spk_embed,
            ref_mels=target,
            f0=f0,
            uv=uv,
            energy=energy,
            infer=infer,
        )
        losses = {}
        if "diff_loss" in output:
            losses["mel"] = output["diff_loss"]
        if not return_output:
            return losses
        else:
            return losses, output

    def _training_step(self, sample, batch_idx):
        log_outputs = self.run_model(self.model, sample)
        total_loss = sum(
            [
                v
                for v in log_outputs.values()
                if isinstance(v, torch.Tensor) and v.requires_grad
            ]
        )
        log_outputs["batch_size"] = sample["hubert"].size()[0]
        log_outputs["lr"] = self.scheduler.get_lr()[0]
        return total_loss, log_outputs

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(
            optimizer, hparams["decay_steps"], gamma=0.5
        )

    def optimizer_step(self, epoch, batch_idx, optimizer):
        if optimizer is None:
            return
        optimizer.step()
        optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step(self.global_step)

    def validation_step(self, sample):
        outputs = {}

        outputs["losses"] = {}

        outputs["losses"], model_out = self.run_model(
            self.model, sample, return_output=True, infer=False
        )

        outputs["total_loss"] = sum(outputs["losses"].values())
        outputs["nsamples"] = sample["nsamples"]
        outputs = utils.tensors_to_scalars(outputs)

        return outputs
