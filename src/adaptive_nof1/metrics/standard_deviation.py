from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation import Simulation

import numpy


class StandardDeviation(Metric):
    def score(self, simulation: Simulation) -> float:
        return numpy.std(simulation.history.to_df()[self.treatment_name])

    def __str__(self) -> str:
        return f"std({self.treatment_name})"
