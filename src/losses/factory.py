from __future__ import annotations

from enum import Enum, auto
from typing import Any

import torch
from torch.nn.modules.loss import BCELoss, CrossEntropyLoss, _Loss


class LossType(Enum):
    CrossEntropy = auto()
    BCELoss = auto()


def create_loss(loss_type: LossType, loss_params: dict[str, Any] | None = None) -> _Loss:
    _loss_types_map: dict[LossType, torch.nn.modules.loss._Loss] = {
        LossType.CrossEntropy: CrossEntropyLoss,
        LossType.BCELoss: BCELoss,
    }

    if loss_params is None:
        return _loss_types_map[loss_type]()
    return _loss_types_map[loss_type](**loss_params)
