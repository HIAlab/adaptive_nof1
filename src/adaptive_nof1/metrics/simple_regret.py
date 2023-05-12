from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation import Simulation


class SimpleRegret(Metric):
    def score(self, simulation: Simulation) -> float:
        return simulation.history.to_df()["outcome"].sum()

    def __str__(self) -> str:
        return "Simple Regret"
