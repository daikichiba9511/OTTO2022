from __future__ import annotations

from typing import Callable, Literal

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

Phase = Literal["train", "valid", "val"]


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hparams: DictConfig,
        create_dataset_fn: Callable[[Phase], torch.utils.data.Dataset],
        batch_size: int = 4,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.create_dataset_fn = create_dataset_fn

    def setup(self, phase: Phase) -> None:
        if phase in {"train", "valid"}:
            self.train_dataset, self.val_dataset = self.create_dataset_fn(phase)
        else:
            self.test_data = self.create_dataset_fn(phase)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
