from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation_data import SimulationData

from scipy.stats import entropy
import numpy


class Entropy(Metric):
    def score(self, data: SimulationData) -> float:
        value, counts = numpy.unique(
            data.history.to_df()[self.outcome_name], return_counts=True
        )
        return entropy(counts)

    def __str__(self) -> str:
        return f"Entropy ({self.outcome_name})"
