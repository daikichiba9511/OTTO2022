from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

from torchmetrics import Accuracy


class MetricsType(Enum):
    Accuracy = "accuracy"

def create_metrics(
    metrics_types: list[MetricsType], metric_params: list[dict[str, Any] | None]
) -> dict[str, Callable]:
    """
    Args:
        metrics_types:
        metric_params:

    Returns:
        metrics: (dict[str, Callable])
    """
    _metrics_types_map: dict[MetricsType, Callable] = {MetricsType.Accuracy: Accuracy}
    metrics: dict[str, Callable] = {}
    for metric_type, metric_param in zip(metrics_types, metric_params):
        if metric_param is None:
            metrics[metric_type.value] = _metrics_types_map[metric_type]()
            continue
        metrics[metric_type.value] = _metrics_types_map[metric_type](**metric_param)
    return metrics
