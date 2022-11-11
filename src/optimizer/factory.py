from __future__ import annotations

from collections.abc import Iterator
from enum import Enum, auto
from typing import Any

import torch


########################
# Optimizer
########################
class OptimizerType(Enum):
    AdamW = auto()


def create_optimizer(
    optimizer_type: OptimizerType, params: Iterator, optim_params: dict[str, Any] | None = None
) -> torch.optim.Optimizer:
    _optimizer_types_map = {OptimizerType.AdamW: torch.optim.AdamW}
    if optim_params is None:
        return _optimizer_types_map[optimizer_type](params)
    return _optimizer_types_map[optimizer_type](params, **optim_params)


########################
# Scheduler
########################
class SchedulerType(Enum):
    CosineAnnealingLR = auto()
    OneCycleLR = auto()


def create_scheduler(
    scheduler_type: SchedulerType, optimizer: torch.optim.Optimizer, scheduler_params: dict[str, Any] | None = None
) -> torch.optim.lr_scheduler._LRScheduler:

    _scheduler_types_map = {
        SchedulerType.CosineAnnealingLR: torch.optim.lr_scheduler.CosineAnnealingLR,
        SchedulerType.OneCycleLR: torch.optim.lr_scheduler.OneCycleLR,
    }
    if scheduler_params is None:
        return _scheduler_types_map[scheduler_type](optimizer)
    return _scheduler_types_map[scheduler_type](optimizer, **scheduler_params)
