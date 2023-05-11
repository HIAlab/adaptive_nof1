from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation import Simulation


class FrequentistOptimalSelection(Metric):
    def score(self, simulation: Simulation) -> float:
        outcome_groupby = (
            simulation.history.to_df().groupby("treatment")["outcome"].mean()
        )
        best_mean = outcome_groupby.idxmin()
        return {True: 1.0, False: 0.0}[best_mean == simulation.model.best_treatment()]

    def __str__(self) -> str:
        return "Optimal Selection"
