from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation import Simulation

from typing import List


class DifferenceBetweenMetric(Metric):
    def __init__(self, minuend, subtrahend):
        self.minuend = minuend
        self.subtrahend = subtrahend

    def score(self, simulation: Simulation) -> List[float]:
        return self.minuend.score(simulation) - self.subtrahend.score(simulation)

    def __str__(self) -> str:
        return f"{self.minuend} - {self.subtrahend}"
