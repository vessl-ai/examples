import os

os.environ["OMP_NUM_THREADS"] = "1"
import random

import numpy as np
import yaml
from preprocessing.data_gen_utils import get_pitch_crepe
from tqdm import tqdm
from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDatasetBuilder


class BinarizationError(Exception):
    pass


BASE_ITEM_ATTRIBUTES = ["txt", "ph", "wav_fn", "tg_fn", "spk_id"]


class BaseBinarizer:
    """
    Base class for data processing.
    1. *process* and *process_data_split*:
        process entire data, generate the train-test split (support parallel processing);
    2. *process_item*:
        process singe piece of data;
    3. *get_pitch*:
        infer the pitch using some algorithm;
    4. *get_align*:
        get the alignment using 'mel2ph' format (see https://arxiv.org/abs/1905.09263).
    5. phoneme encoder, voice encoder, etc.

    Subclasses should define:
    1. *load_metadata*:
        how to read multiple datasets from files;
    2. *train_item_names*, *valid_item_names*, *test_item_names*:
        how to split the dataset;
    3. load_ph_set:
        the phoneme set.
    """

    def __init__(self, item_attributes=BASE_ITEM_ATTRIBUTES):
        self.binarization_args = hparams["binarization_args"]

        self.items = {}
        # every item in self.items has some attributes
        self.item_attributes = item_attributes

        self.load_meta_data()

        assert all(
            [
                attr in self.item_attributes
                for attr in list(self.items.values())[0].keys()
            ]
        )
        self.item_names = sorted(list(self.items.keys()))

        if self.binarization_args["shuffle"]:
            random.seed(1234)
            random.shuffle(self.item_names)

        # set default get_pitch algorithm
        if hparams["use_crepe"]:
            self.get_pitch_algorithm = get_pitch_crepe
        else:
            raise NotImplementedError

    def load_meta_data(self):
        raise NotImplementedError

    @property
    def train_item_names(self):
        raise NotImplementedError

    @property
    def valid_item_names(self):
        raise NotImplementedError

    @property
    def test_item_names(self):
        raise NotImplementedError

    def _phone_encoder(self):
        """
        use hubert encoder
        """
        raise NotImplementedError

    def load_ph_set(self, ph_set):
        raise NotImplementedError

    def meta_data_iterator(self, prefix):
        if prefix == "valid":
            item_names = self.valid_item_names
        elif prefix == "test":
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            meta_data = self.items[item_name]
            yield item_name, meta_data

    def process(self):
        exp_name = hparams["exp_name"]
        os.makedirs(f"./raw/{exp_name}/binary", exist_ok=True)
        self.phone_encoder = self._phone_encoder()
        self.process_data_split("valid")
        self.process_data_split("test")
        self.process_data_split("train")

    def process_data_split(self, prefix):
        exp_name = hparams["exp_name"]
        data_dir = f"./raw/{exp_name}/binary"
        args = []
        builder = IndexedDatasetBuilder(f"{data_dir}/{prefix}")
        lengths = []
        f0s = []
        total_sec = 0
        for item_name, meta_data in self.meta_data_iterator(prefix):
            args.append([item_name, meta_data, self.binarization_args])
        spec_min = []
        spec_max = []
        # code for single cpu processing
        for i in tqdm(reversed(range(len(args))), total=len(args)):
            a = args[i]
            item = self.process_item(*a)
            if item is None:
                continue
            spec_min.append(item["spec_min"])
            spec_max.append(item["spec_max"])
            builder.add_item(item)
            lengths.append(item["len"])
            total_sec += item["sec"]

        if prefix == "train":
            spec_max = np.max(spec_max, 0)
            spec_min = np.min(spec_min, 0)

            with open(hparams["config_path"], encoding="utf-8") as f:
                # This requires config.yml to exists originally
                _hparams = yaml.safe_load(f)
                _hparams["spec_max"] = spec_max.tolist()
                _hparams["spec_min"] = spec_min.tolist()
            with open(hparams["config_path"], "w", encoding="utf-8") as f:
                yaml.safe_dump(_hparams, f)
        builder.finalize()
        np.save(f"{data_dir}/{prefix}_lengths.npy", lengths)
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(
                f"{data_dir}/{prefix}_f0s_mean_std.npy",
                [np.mean(f0s).item(), np.std(f0s).item()],
            )
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    def process_item(self, item_name, meta_data, binarization_args):
        from preprocessing.process_pipeline import File2Batch

        return File2Batch.temporary_dict2processed_input(
            item_name, meta_data, self.phone_encoder, binarization_args
        )

    def get_align(self, meta_data, mel, phone_encoded, res):
        raise NotImplementedError


if __name__ == "__main__":
    set_hparams()
    BaseBinarizer().process()
