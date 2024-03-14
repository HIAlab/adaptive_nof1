from __future__ import annotations
from re import S
from typing import List

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation_data import SimulationData
from adaptive_nof1.helpers import flatten_dictionary

import numpy
import random


# The Simple Regret is defined as the Expectation of the difference between the best arm and the identified arm
class SimpleRegret(Metric):
    def score(self, data: SimulationData) -> List[float]:
        assert (
            "expectations_of_interventions" in data.additional_config
        ), "Simple Regret can only be calculated if the expectations of interventions are known"
        df = data.history.to_df()
        recommenddations = df["current_best_arm"]
        raise NotImplementedError

    def __str__(self) -> str:
        return "Simple Regret"


# The Simple Regret is defined as the Expectation of the difference between the best arm and the identified arm
# In this version, the best arm is chosen by the highest mean reported by the inference model in the debug data
class SimpleRegretWithMean(SimpleRegret):
    def best_arm_per_timestep(self, data: SimulationData):
        debug_data = data.history.debug_data()
        best_arms_per_timestep = []
        prev_value = None

        for d in debug_data:
            if "mean" in d:
                prev_value = d["mean"]
                max_value = numpy.max(prev_value)
                max_indices = numpy.where(prev_value == max_value)[0]
                # Randomly select one of these indices
                best_arms_per_timestep.append(random.choice(max_indices))
            else:
                best_arms_per_timestep.append(best_arms_per_timestep[-1])

        return best_arms_per_timestep

    def score(self, data: SimulationData) -> List[float]:
        assert (
            "expectations_of_interventions" in data.additional_config
        ), "Simple Regret can only be calculated if the expectations of interventions are known"
        expectations_per_arm = data.additional_config["expectations_of_interventions"]

        best_arm_expectation = max(expectations_per_arm)

        expectations_per_timestep = [
            expectations_per_arm[arm] for arm in self.best_arm_per_timestep(data)
        ]

        return best_arm_expectation - numpy.array(expectations_per_timestep)

    def __str__(self) -> str:
        return "Simple Regret With Mean"


class SimpleRegretWithBayesian(SimpleRegretWithMean):
    def best_arm_per_timestep(self, data: SimulationData):
        debug_data = data.history.debug_data()
        best_arms_per_timestep = []
        prev_value = None

        for d in debug_data:
            if "upper_bounds_array" in d:
                prev_value = d["upper_bounds_array"]
                max_value = numpy.max(prev_value)
                max_indices = numpy.where(prev_value == max_value)[0]
                # Randomly select one of these indices
                best_arms_per_timestep.append(random.choice(max_indices))
            elif "probabilities" in d:
                prev_value = d["probabilities"]
                max_value = numpy.max(prev_value)
                max_indices = numpy.where(prev_value == max_value)[0]
                # Randomly select one of these indices
                best_arms_per_timestep.append(random.choice(max_indices))
            else:
                best_arms_per_timestep.append(best_arms_per_timestep[-1])

        return best_arms_per_timestep

    def __str__(self) -> str:
        return "Simple Regret With Bayesian Model"
