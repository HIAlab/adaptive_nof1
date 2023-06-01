from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation import Simulation

from ..policies import ConstantPolicy

import numpy


class RegretAgainstConstantPolicy(Metric):
    def __init__(self, constant_action):
        self.constant_action = constant_action

    def score(self, simulation: Simulation):
        # Todo: Test that this is 0 if using same constant policy
        constant_simulation = Simulation.from_model_and_policy_with_copy(
            simulation.model,
            ConstantPolicy(
                number_of_actions=simulation.policy.number_of_actions,
                action=self.constant_action,
            ),
        )
        for _ in range(len(simulation.history)):
            constant_simulation.step()

        constant_df = constant_simulation.history.to_df()
        return numpy.cumsum(
            constant_df[self.outcome_name]
            - simulation.history.to_df()[self.outcome_name]
        )

    def __str__(self) -> str:
        return f"RegretAgainstConstantAction()"
