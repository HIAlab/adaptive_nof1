from src.bayes import BayesianInference
import pytest

import pandas as pd
import numpy as np
import xarray as xr


@pytest.fixture
def simple_trace():
    chain_one_data = [[1, 2] for _ in range(100)]
    chain_two_data = [[2, 1] for _ in range(100)]
    data = [chain_one_data, chain_two_data]
    return {
        "posterior": {
            "average_treatment_effect": xr.Dataset(
                data_vars=dict(
                    average_treatment_effect=(
                        ["chain", "draw", "average_treatment_effect_dim_0"],
                        data,
                    ),
                ),
                coords=dict(
                    chain=range(2),
                    draw=range(100),
                    average_treatment_effect_dim_0=range(2),
                ),
                attrs=dict(description="Simple Trace Mockup"),
            )
        }
    }


def test_approximate_max_probabilities(simple_trace):
    inference = BayesianInference()
    inference.trace = simple_trace

    assert list(inference.approximate_max_probabilities()) == [0.5, 0.5]
