from adaptive_nof1.helpers import *
from adaptive_nof1.metrics import RegretAgainstOtherConfiguration
from adaptive_nof1 import SimulationData

from .fixtures import simple_history

import pytest
from frozendict import frozendict


# The metric should only compare the middle timepoint, since this is the only one with t == 1 present in both
def test_matching_of_timepoints(simple_history):
    first_two = SimulationData(
        history=simple_history[0:2],
        model="FirstTwo",
        policy="TestPolicy",
        patient_id=0,
    )
    last_two = SimulationData(
        history=simple_history[1:3],
        model="LastTwo",
        policy="TestPolicy",
        patient_id=0,
    )

    def switch_config(config):
        config = config.copy()
        config["model"] = "LastTwo"
        return config

    metric = RegretAgainstOtherConfiguration(
        configuration_transform_function=switch_config,
        config_to_simulation_data={frozendict(last_two.configuration): last_two},
    )
    score = metric.score(first_two)
    assert len(score) == 1
    assert score[0] == 0
