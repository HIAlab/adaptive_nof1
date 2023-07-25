from adaptive_nof1.inference.bayes import GaussianAverageTreatmentEffect
import pytest

import pandas as pd
import numpy as np

from adaptive_nof1.models.pill_model import PillModel
from adaptive_nof1.policies.combined_policy import CombinedPolicy
from adaptive_nof1.series_of_simulations_runner import SeriesOfSimulationsRunner
from adaptive_nof1.series_of_simulations_data import SeriesOfSimulationsData
from adaptive_nof1.policies.fixed_policy import FixedPolicy
from .fixtures import runner, fixed_policy


def test_data_idempotency(runner):
    series = runner.simulate(100)
    serialized = series.serialize()
    reconstructed_series = SeriesOfSimulationsData.deserialize(serialized)

    assert len(reconstructed_series.simulations) != 0
    assert len(reconstructed_series.simulations) == len(series.simulations)

    assert series == reconstructed_series
