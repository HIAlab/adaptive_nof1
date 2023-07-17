from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric

import numpy


class StandardDeviation(Metric):
    def score(self, data: SimulationData) -> float:
        return numpy.std(data.history.to_df()[self.treatment_name])

    def __str__(self) -> str:
        return f"std({self.treatment_name})"
