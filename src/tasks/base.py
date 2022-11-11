from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Optional, Protocol

import pytorch_lightning as pl
import torch
import torch.nn as nn
from loguru import logger
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


def train_one_step(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: torch.nn.modules.loss._Loss
) -> tuple[_Loss, torch.Tensor]:
    output = model(x)
    loss = criterion(output, y)
    return loss, output


class IMetrics(Protocol):
    def __call__(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        ...


@torch.no_grad()
def valid_one_step(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, metrics: dict[str, IMetrics]
) -> dict[str, torch.Tensor]:
    output = model(x)
    metrics_output = {metric_name: metric(output, y) for metric_name, metric in metrics.items()}
    metrics_output.update({"preds": output})
    return metrics_output


@torch.inference_mode()
def test_one_step(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = model(x)
    return output


def test(model: nn.Module, dataloader: DataLoader) -> dict[str, list[torch.Tensor]]:
    results = []
    for x in dataloader:
        result = test_one_step(model, x)
        results.append(result)
    return {"results": results}


def training(
    config: DictConfig,
    lit_module: pl.LightningModule,
    data_module: pl.LightningDataModule,
    config_name: str,
    model_name: str,
    datamodule_name: str,
    model_path: Path,
    max_epochs: int,
    monitor_metric: str,
    monitor_mode: Literal["min", "max"],
    wandb_project: str,
    precision: Literal[16, 32] = 16,
    patience: int = 3,
    fold: int = 0,
    seed: int = 42,
    debug: bool = False,
    accumulation_steps: int = 1,
    save_weights_only: bool = False,
    save_last: Optional[bool] = None,
    checkpoint_path: Optional[str] = None,
    log_dir: str = "wandb",
    num_sanity_val_steps: int = 3,
    gradient_clip_val: Optional[float] = None,
) -> None:
    exp_version = config_name + "_" + model_name

    pl.seed_everything(seed=seed)
    logger.info(f" ########## Fold: {fold} , seed: {seed}, exp_version: {exp_version} ############ ")

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        filename=model_name + "-{epoch}-{step}",
        verbose=True,
        save_top_k=1,
        save_weights_only=save_weights_only,
        save_last=save_last,
        mode=monitor_mode,
    )
    early_stop_callback = EarlyStopping(monitor=monitor_metric, patience=patience)
    lr_monitor = LearningRateMonitor()
    wandb_logger = WandbLogger(
        name=config_name + "_" + model_name + "_" + f"fold{fold}",
        save_dir=log_dir,
        project=wandb_project,
        version=hashlib.sha224(bytes(str(dict(config)), "utf8")).hexdigest()[:4],
        anonymous=True,
        group=config_name + "_" + model_name,
    )

    params = {
        "logger": wandb_logger,
        "max_epochs": max_epochs,
        "checkpoint_callback": checkpoint_callback,
        "callbacks": [early_stop_callback, checkpoint_callback, lr_monitor],
        "accumulate_grad_batches": accumulation_steps,
        # "amp_backend": "native",
        # "amp_level": "",
        "precision": precision,
        "gpus": 1,
        "accelerator": None,
        "limit_train_batches": 1.0 if not debug else 0.05,
        "check_val_every_n_epoch": 1,
        "limit_val_batches": 1.0 if not debug else 0.05,
        "limit_test_batches": 0.0,
        "num_sanity_val_steps": num_sanity_val_steps,  # OneCycleLRのときは0にする
        # "num_nodes": 1,
        "gradient_clip_val": gradient_clip_val,
        "log_every_n_steps": 10,
        "flush_logs_every_n_steps": 10,
        "profiler": "simple",
        "deterministic": False,
        "resume_from_checkpoint": checkpoint_path,
        "weights_summary": "top",
        "reload_dataloaders_every_epoch": True,
        # "replace_sampler_ddp": False,
    }

    trainer = pl.Trainer(**params)
    trainer.fit(lit_module, datamodule=data_module)
