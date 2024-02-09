from adaptive_nof1.helpers import *
from adaptive_nof1.metrics import RegretAgainstOtherConfiguration
from adaptive_nof1 import SimulationData
from adaptive_nof1.metrics.simple_regret import SimpleRegret, SimpleRegretWithMean

from .fixtures import simple_history

import pytest
from frozendict import frozendict


def test_extraction_of_best_arm(simple_history):
    metric = SimpleRegretWithMean()

    data = SimulationData(
        history=simple_history,
        additional_config={"expectations_of_interventions": [1, 2, 3]},
        model="TestModel",
        policy="TestPolicy",
        patient_id=0,
    )
    best_arms = metric.best_arm_per_timestep(data)
    assert len(best_arms) == 3
    assert list(best_arms) == [0, 1, 1]


# In this test case, the best arm is arm 2 with a mean of 3.
# Therefore, the identified best arms (0 and 1) have a simple regret of 2 and 1 respectively.
def test_calculation_of_score(simple_history):
    metric = SimpleRegretWithMean()

    data = SimulationData(
        history=simple_history,
        additional_config={"expectations_of_interventions": [1, 2, 3]},
        model="TestModel",
        policy="TestPolicy",
        patient_id=0,
    )
    score = metric.score(data)
    assert len(score) == 3
    assert list(score) == [2, 1, 1]
