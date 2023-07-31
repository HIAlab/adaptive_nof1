from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation_data import SimulationData


class FrequentistOptimalSelection(Metric):
    def score(self, data: SimulationData) -> float:
        outcome_groupby = data.history.to_df().groupby("treatment")["outcome"].mean()
        best_mean = outcome_groupby.idxmin()
        return {True: 1.0, False: 0.0}[best_mean == data.model.best_treatment()]

    def __str__(self) -> str:
        return "Optimal Selection"
