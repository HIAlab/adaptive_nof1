from __future__ import annotations
from re import S
from typing import List

from adaptive_nof1.metrics.metric import Metric
from adaptive_nof1.simulation_data import SimulationData
from adaptive_nof1.helpers import flatten_dictionary

import numpy
import random


class BestArmIdentification(Metric):
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

        return best_arms_per_timestep

    def score(self, data: SimulationData) -> List[float]:
        assert (
            "expectations_of_interventions" in data.additional_config
        ), "BestArmIdentification can only be calculated if the expectations of interventions are known"
        expectations_per_arm = data.additional_config["expectations_of_interventions"]

        best_arm_index = numpy.argmax(expectations_per_arm)

        return self.best_arm_per_timestep(data) == best_arm_index

    def __str__(self) -> str:
        return "Best Arm Identification With Mean"
