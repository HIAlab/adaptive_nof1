from __future__ import annotations

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.models.model import Model
from adaptive_nof1.simulation_data import SimulationData
from adaptive_nof1.simulation_runner import SimulationRunner
from frozendict import frozendict

from ..policies import ConstantPolicy
from ..policies import Policy

import numpy


class RegretAgainstOtherConfiguration(Metric):
    def __init__(
        self,
        config_to_simulation_data: dict[dict, SimulationData],
        configuration_transform_function,
        name="",
        **kwargs,
    ):
        self.config_to_simulation_data = config_to_simulation_data
        self.configuration_transform_function = configuration_transform_function
        self.name = name
        super().__init__(**kwargs)

    def score(self, data: SimulationData):
        config_to_compare_against = str(
            self.configuration_transform_function(data.configuration)
        )
        assert (
            config_to_compare_against in self.config_to_simulation_data
        ), f"Can not calculate compare metric, desired config not present in dataset\nTried to fetch {config_to_compare_against}"

        counterfactual_df = self.config_to_simulation_data[
            config_to_compare_against
        ].history.to_df()
        data_df = data.history.to_df()

        merge = data_df.merge(
            counterfactual_df,
            how="inner",
            validate="one_to_one",
            on="t",
            suffixes=(None, "_counterfactual"),
        )
        return numpy.cumsum(
            merge[self.outcome_name + "_counterfactual"] - merge[self.outcome_name]
        )

    def __str__(self) -> str:
        return f"RegretAgainstOtherConfiguration({self.name})"


class RegretAgainstPolicy(Metric):
    def __init__(self, policy: Policy, model: Model, **kwargs):
        self.policy = policy
        self.model = model
        super().__init__(**kwargs)

    def score(self, data: SimulationData):
        # Todo: Test that this is 0 if using same constant policy
        counterfactual_simulation = SimulationRunner.from_model_and_policy_with_copy(
            self.model,
            self.policy,
        )
        counterfactual_simulation.model.reset()
        for _ in range(len(data.history)):
            counterfactual_simulation.step()

        counterfactual_df = counterfactual_simulation.history.to_df()
        return numpy.cumsum(
            counterfactual_df[self.outcome_name]
            - data.history.to_df()[self.outcome_name]
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
