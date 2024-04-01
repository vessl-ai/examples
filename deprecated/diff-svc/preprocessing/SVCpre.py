import logging
from copy import deepcopy

from preprocessing.base_binarizer import BaseBinarizer
from preprocessing.process_pipeline import File2Batch
from utils.hparams import hparams

SVCSINGING_ITEM_ATTRIBUTES = ["wav_fn"]


class SVCBinarizer(BaseBinarizer):
    def __init__(self, item_attributes=SVCSINGING_ITEM_ATTRIBUTES):
        super().__init__(item_attributes)
        self.item_names = sorted(list(self.items.keys()))
        self._train_item_names, self._test_item_names = self.split_train_test_set(
            self.item_names
        )

    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)

        # FIXME: Use needs to be able to change this. Depricated manual test selection.
        # If not manual, use 5 elements for validation.
        test_item_names = item_names[-5:]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    @property
    def test_item_names(self):
        return self._test_item_names

    def load_meta_data(self):
        self.items = File2Batch.file2temporary_dict()

    def _phone_encoder(self):
        from preprocessing.hubertinfer import Hubertencoder

        return Hubertencoder(hparams["hubert_path"])
