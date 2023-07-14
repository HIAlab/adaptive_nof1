from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation import Simulation

from ..policies import ConstantPolicy
from ..policies import Policy

import numpy


class RegretAgainstPolicy(Metric):
    def __init__(self, policy: Policy, **kwargs):
        self.policy = policy
        super().__init__(**kwargs)

    def score(self, simulation: Simulation):
        # Todo: Test that this is 0 if using same constant policy
        counterfactual_simulation = Simulation.from_model_and_policy_with_copy(
            simulation.model,
            self.policy,
        )
        counterfactual_simulation.model.reset()
        for _ in range(len(simulation.history)):
            counterfactual_simulation.step()

        counterfactual_df = counterfactual_simulation.history.to_df()
        return numpy.cumsum(
            counterfactual_df[self.outcome_name]
            - simulation.history.to_df()[self.outcome_name]
        )

    def __str__(self) -> str:
        return f"RegretAgainstPolicy{self.policy}"


class RegretAgainstConstantPolicy(RegretAgainstPolicy):
    def __init__(self, constant_action, **kwargs):
        self.constant_action = constant_action
        self.policy = ConstantPolicy(
            action=self.constant_action,
            number_of_actions=self.constant_action,
        )
        # super().__init__(**kwargs)

    def __str__(self) -> str:
        return f"RegretAgainstConstantAction({self.constant_action})"
