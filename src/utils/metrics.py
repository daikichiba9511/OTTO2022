from __future__ import annotations

import math
from typing import TypedDict


class AvgMeterDict(TypedDict):
    name: str
    avg: float
    row_values: list[int | float]


class AverageMeter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def __str__(self) -> str:
        return f"Metrics {self.name}: Avg {self.avg}, Row values {self.rows}"

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.rows: list[float | int] = []

    def update(self, value: float | int, n: int = 1) -> None:
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.rows.append(value)

    def to_dict(self) -> AvgMeterDict:
        return {"name": self.name, "avg": self.avg, "row_values": self.rows}


if __name__ == "__main__":
    avg_meter = AverageMeter("test")
    avg_meter.update(10)
    avg_meter.update(20)
    avg_meter.update(30)
    print(avg_meter)

    assert math.isclose(avg_meter.avg, 20.0)

    avg_dict = avg_meter.to_dict()
    assert avg_dict["name"] == "test"
    assert math.isclose(avg_dict["avg"], 20.0)
    assert avg_dict["row_values"] == [10, 20, 30]
