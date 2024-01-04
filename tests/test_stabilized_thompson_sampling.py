from adaptive_nof1.helpers import array_almost_equal
from adaptive_nof1.policies import stabilized_thompson_sampling
from adaptive_nof1.policies.balanced_thompson_sampling import BalancedThompsonSampling
from adaptive_nof1.policies.stabilized_thompson_sampling import StabilizedThompsonSampling

import pytest

import pandas as pd
import numpy as np
import xarray as xr

from tests.helper_functions import outcomes_to_history
from tests.mocks import MockBayesianModel

def test_stabilization():
    stabilized_thompson_sampling = StabilizedThompsonSampling(
        inference_model=MockBayesianModel(max_probabilities=[]),
        number_of_actions=2,
        length=10,
    )

    assert stabilized_thompson_sampling.stabilize_probabilities([0, 1], 0) == [0.5, 0.5]
    assert stabilized_thompson_sampling.stabilize_probabilities([0.5, 0.5], 1) == [0.5, 0.5]
    assert array_almost_equal(stabilized_thompson_sampling.stabilize_probabilities([0.8, 0.2], 0.5), [0.66666, 0.33333], epsilon=0.001)

