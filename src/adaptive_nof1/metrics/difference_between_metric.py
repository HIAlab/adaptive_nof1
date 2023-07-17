from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric

from typing import List

from adaptive_nof1.simulation_data import SimulationData


class DifferenceBetweenMetric(Metric):
    def __init__(self, minuend, subtrahend):
        self.minuend = minuend
        self.subtrahend = subtrahend

    def score(self, data: SimulationData) -> List[float]:
        return self.minuend.score(data) - self.subtrahend.score(data)

    def __str__(self) -> str:
        return f"{self.minuend} - {self.subtrahend}"
