from adaptive_nof1.models.self_experimentation_model import SelfExperimentationModel
from adaptive_nof1.policies import ConstantPolicy
from adaptive_nof1 import SimulationRunner
import pytest


def test_self_experimentation_model_baseline_1():
    runner = SimulationRunner.from_model_and_policy(
        model=SelfExperimentationModel(
            patient_id=0,
            intervention_effects=[0],
            baseline_model="linear",
            baseline_config={"start": 2, "end": 12, "min": 0, "max": 10},
        ),
        policy=ConstantPolicy(action=0, number_of_actions=1),
    )
    data = runner.simulate(15)
    expected = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10]
    assert list(data.history.to_df()["baseline"]) == expected


def test_self_experimentation_model_baseline_2():
    runner = SimulationRunner.from_model_and_policy(
        model=SelfExperimentationModel(
            patient_id=0,
            intervention_effects=[0],
            baseline_model="linear",
            baseline_config={"start": 5, "end": 10, "min": 0.5, "max": 1.0},
        ),
        policy=ConstantPolicy(action=0, number_of_actions=1),
    )
    data = runner.simulate(15)
    expected = [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert list(data.history.to_df()["baseline"]) == expected
