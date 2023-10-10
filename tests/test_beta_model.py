from adaptive_nof1.inference.beta_model import BetaModel
from adaptive_nof1.basic_types import History, Observation
import pytest

from .fixtures import simple_history

from adaptive_nof1.helpers import array_almost_equal

import pandas as pd
import numpy as np
import xarray as xr


def test_dataframe_probability_transformation(simple_history):
    model = BetaModel()
    p = model.dataframe_to_n_successes(simple_history.to_df(), number_of_treatments=3)
    assert p == [0, 1, 1]


def test_dataframe_n_trials_transformation(simple_history):
    model = BetaModel()
    p = model.dataframe_to_n_trials(simple_history.to_df(), number_of_treatments=3)
    assert p == [1, 1, 1]


def test_dataframe_n_trials_transformation_2():
    history = History(
        observations=[
            Observation(
                context={"activity": 20},
                treatment={"treatment": 1},
                outcome={"outcome": 3},
                t=1,
                patient_id=0,
            )
        ]
    )
    model = BetaModel()
    p = model.dataframe_to_n_trials(history.to_df(), number_of_treatments=2)
    assert p == [0, 1]


# Same tests are in the self-e app:
def outcomes_to_history(treatments, outcomes):
    assert len(treatments) == len(outcomes)
    return History(
        observations=[
            Observation(
                context={},
                treatment={"treatment": treatment},
                outcome={"outcome": outcome},
                t=t,
                patient_id=0,
            )
            for treatment, outcome, t in zip(
                treatments, outcomes, range(len(treatments))
            )
        ]
    )


@pytest.mark.sampling
@pytest.mark.parametrize(
    "treatments, outcomes, expected",
    [
        ([0, 0, 1, 1], [2, 4, 3, 3], [0.8, 0.2]),
        ([0, 1], [3, 3], [0.5, 0.5]),
        ([0, 1, 2], [3, 3, 3], [0.33, 0.33, 0.33]),
        ([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 5, 4, 3, 2, 1], [0.5, 0.5]),
    ],
)
def test_beta_model(treatments, outcomes, expected):
    number_of_treatments = max(treatments) + 1
    model = BetaModel()
    model.update_posterior(
        outcomes_to_history(treatments, outcomes), number_of_treatments
    )
    p = model.approximate_max_probabilities(number_of_treatments, {"t": len(outcomes)})
    assert array_almost_equal(list(p), expected, epsilon=0.02)


def test_beta_model_empty_history():
    number_of_treatments = 3
    model = BetaModel()
    model.update_posterior(outcomes_to_history([], []), number_of_treatments)
    p = model.approximate_max_probabilities(number_of_treatments, {"t": 0})
    assert array_almost_equal(list(p), [0.33, 0.33, 0.33], epsilon=0.02)
