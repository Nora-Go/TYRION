from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from model.utils import *


class GenericDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_config,
                 validation_config,
                 test_config,
                 custom_collate,
                 batch_size=1,
                 n_workers=8,
                 stage="train",
                 persistent_workers=True,
                 **kwargs):
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.persistent_workers = persistent_workers
        self.stage = stage

        if self.stage != "test":
            self.train_data = instantiate_from_config(train_config, **kwargs)
            self.val_data = instantiate_from_config(validation_config, **kwargs)
        else:
            self.test_data = instantiate_from_config(test_config, **kwargs)

        if custom_collate is not None:
            self.cc = lambda x: instantiate_from_config_function(custom_collate)(x)
        else:
            self.cc = lambda x: custom_collate(x)

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(self.train_data, collate_fn=self.cc, batch_size=self.batch_size,
                          num_workers=self.n_workers, shuffle=True, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, collate_fn=self.cc, batch_size=1,
                          num_workers=self.n_workers, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, collate_fn=self.cc, batch_size=1,
                          num_workers=self.n_workers, persistent_workers=self.persistent_workers)

    def teardown(self, stage: Optional[str] = None):
        pass
