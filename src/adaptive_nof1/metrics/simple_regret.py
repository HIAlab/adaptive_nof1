from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation import Simulation

import numpy


class SimpleRegret(Metric):
    def score(self, simulation: Simulation) -> float:
        return numpy.cumsum(-simulation.history.to_df()[self.outcome_name])

    def __str__(self) -> str:
        return "Simple Regret"

class BestCaseSimpleRegret(Metric):
    def score(self, simulation: Simulation) -> float:
        counterfactual_max = simulation.history.counterfactual_outcomes_df(self.outcome_name).max(axis=1)
        return numpy.cumsum(-counterfactual_max)

    def __str__(self) -> str:
        return "BestCaseSimpleRegret"
