import pytest

import pandas as pd
import numpy as np
import xarray as xr

from tests.mocks import MockBayesianModel

from tests.helper_functions import outcomes_to_history


def test_sequential_halving(treatments, probabilities):
    sequential_halving = SequentialHalving(
        inference_model=MockBayesianModel(max_probabilities=probabilities),
        length=16,
    )
    sequential_halving.number_of_interventions = 4

    mock_outcomes = [0] * len(treatments)
    mock_context = {}

    list_of_actions = []
    for _ in range(16):
        list_of_actions.append(
            sequential_halving.choose_action(
                outcomes_to_history(treatments, mock_outcomes),
                mock_context,
            )["treatment"]
        )

    assert list_of_actions == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
