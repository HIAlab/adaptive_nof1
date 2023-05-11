from __future__ import annotations

from src.adaptive_nof1.metrics.metric import Metric
from src.adaptive_nof1.simulation import Simulation


class SimpleRegret(Metric):
    def score(self, simulation: Simulation) -> float:
        return simulation.history.to_df()["outcome"].mean()

    def __str__(self) -> str:
        return "Simple Regret"
