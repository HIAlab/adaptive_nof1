from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation import Simulation

import numpy


class SimpleRegret(Metric):
    def score(self, simulation: Simulation) -> float:
        return numpy.cumsum(-simulation.history.to_df()["outcome"])

    def __str__(self) -> str:
        return "Simple Regret"
