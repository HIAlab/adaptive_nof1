from __future__ import annotations
from typing import List

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation_data import SimulationData

import numpy


class SimpleRegret(Metric):
    def score(self, data: SimulationData) -> List[float]:
        return numpy.cumsum(-data.history.to_df()[self.outcome_name])

    def __str__(self) -> str:
        return "Simple Regret"


class MaximizingSimpleRegret(Metric):
    def score(self, data: SimulationData) -> List[float]:
        return numpy.cumsum(data.history.to_df()[self.outcome_name])

    def __str__(self) -> str:
        return "Simple Regret"


class BestCaseSimpleRegret(Metric):
    def score(self, data: SimulationData) -> float:
        counterfactual_max = data.history.counterfactual_outcomes_df(
            self.outcome_name
        ).max(axis=1)
        return numpy.cumsum(-counterfactual_max)

    def __str__(self) -> str:
        return "BestCaseSimpleRegret"
