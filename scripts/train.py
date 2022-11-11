import hashlib
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from typing_extensions import Literal

from configs.factory import get_config
from utils.parse import parse_args


# TODO: model_nameを見直す
def run(
    config: DictConfig,
    fold: int,
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

    lit_module = getattr(task_path, model_name)(config)
    data_module = getattr(task_path, datamodule_name)(config)
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


def main():
    parsed_args = parse_args()
    config_name = Path(parsed_args.config)
    config = get_config(config_name)
    for fold in range(config.n_splits):
        if (config.debug and fold > 1) or (fold not in config.train_fold):
            continue
        run(
            config_name=str(config_name),
            config=config,
            fold=fold,
            model_name=config.model_name,
            datamodule_name=config.datamodule_name,
            model_path=config.model_path,
            max_epochs=config.max_epochs,
            monitor_metric=config.callbacks.monitor_metric,
            monitor_mode=config.callbacks.monitor_mode,
            wandb_project=config.wandb_project,
        )


if __name__ == "__main__":
    main()
