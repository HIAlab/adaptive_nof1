from adaptive_nof1.policies.balanced_thompson_sampling import BalancedThompsonSampling
import pytest

import pandas as pd
import numpy as np
import xarray as xr

from tests.mocks import MockBayesianModel

from tests.helper_functions import outcomes_to_history


class MockBayesianModel:
    def __init__(
        self,
        treatment_name="treatment",
        outcome_name="outcome",
        max_probabilities=[0.5, 0.5],
        upper_confidence_bouns=[1, 1],
    ):
        self.trace = None
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self.max_probabilities = max_probabilities
        self.upper_confidence_bounds = upper_confidence_bouns

    def get_upper_confidence_bounds(self, variable_name, epsilon: float = 0.05):
        return self.upper_confidence_bounds

    def approximate_max_probabilities(self, number_of_treatments, context):
        return self.max_probabilities

    def update_posterior(self, history, number_of_treatments):
        pass


@pytest.mark.parametrize(
    "treatments, probabilities, next_expected_action",
    [
        ([0], [0.5, 0.5], 1),
        ([1], [0.5, 0.5], 0),
        ([0, 0, 0, 1, 1], [0.8, 0.2], 0),
        ([0, 0, 0, 0, 0], [0.8, 0.2], 1),
        ([0, 1, 0, 1], [0.8, 0.2], 0),
    ],
)
def test_balanced_thompson_sampling(treatments, probabilities, next_expected_action):
    balanced_thompson_sampling = BalancedThompsonSampling(
        inference_model=MockBayesianModel(max_probabilities=probabilities),
        number_of_actions=2,
    )

    mock_outcomes = [0] * len(treatments)
    mock_context = {}

    next_action = balanced_thompson_sampling.choose_action(
        outcomes_to_history(treatments, mock_outcomes),
        mock_context,
    )

    assert next_action["treatment"] == next_expected_action
