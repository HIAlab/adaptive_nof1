from adaptive_nof1.inference.bayes import GaussianAverageTreatmentEffect
import pytest

import pandas as pd
import numpy as np

from adaptive_nof1.models.pill_model import PillModel
from adaptive_nof1.policies.combined_policy import CombinedPolicy
from adaptive_nof1.series_of_simulations import SeriesOfSimulations
from adaptive_nof1.policies.fixed_policy import FixedPolicy

import pickle

NUMBER_OF_ACTIONS = 5
NUMBER_OF_PATIENTS = 5


@pytest.fixture
def series():
    return SeriesOfSimulations(
        model_from_patient_id= lambda patient_id: PillModel(patient_id),
        n_patients=NUMBER_OF_PATIENTS,
        policy=CombinedPolicy(
            [
                FixedPolicy(number_of_actions=NUMBER_OF_ACTIONS),
            ],
            number_of_actions=NUMBER_OF_ACTIONS,
        ),
        length=100,
    )


def test_idempotency(series):
    pickled_series = pickle.dumps(series)
    reconstructed_series = pickle.loads(pickled_series)
    
    assert len(reconstructed_series.simulations) != 0
    assert len(reconstructed_series.simulations) == len(series.simulations)

    for simulation, reconstructed_simulation in zip(series.simulations, reconstructed_series.simulations):
        simulation.step()
        reconstructed_simulation.step()
        
        assert simulation.history == reconstructed_simulation.history
