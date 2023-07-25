from adaptive_nof1.inference.bayes import GaussianAverageTreatmentEffect
import pytest

import pandas as pd
import numpy as np

from adaptive_nof1.models.pill_model import PillModel
from adaptive_nof1.policies.combined_policy import CombinedPolicy
from adaptive_nof1.series_of_simulations_runner import SeriesOfSimulationsRunner
from adaptive_nof1.series_of_simulations_data import SeriesOfSimulationsData
from adaptive_nof1.policies.fixed_policy import FixedPolicy

import pickle

NUMBER_OF_ACTIONS = 5
NUMBER_OF_PATIENTS = 5


@pytest.fixture
def runner():
    runner = SeriesOfSimulationsRunner(
        model_from_patient_id=lambda patient_id: PillModel(patient_id=patient_id),
        n_patients=NUMBER_OF_PATIENTS,
        policy=CombinedPolicy(
            [
                FixedPolicy(number_of_actions=NUMBER_OF_ACTIONS),
            ],
            number_of_actions=NUMBER_OF_ACTIONS,
        ),
    )
    return runner


def test_data_idempotency(runner):
    series = runner.simulate(100)
    serialized = series.serialize()
    reconstructed_series = SeriesOfSimulationsData.deserialize(serialized)

    assert len(reconstructed_series.simulations) != 0
    assert len(reconstructed_series.simulations) == len(series.simulations)

    assert series == reconstructed_series
