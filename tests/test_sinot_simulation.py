from _pytest.fixtures import FixtureFunction, fixture
from adaptive_nof1.inference.bayes import GaussianAverageTreatmentEffect
import pytest
from adaptive_nof1.simulation import SinotModel

import pandas as pd
import numpy as np
import xarray as xr

import itertools


@fixture
def parameter_file_path():
    return "sinot/example/example_params.json"


@fixture
def treatment_schedule():
    return [1] * 40 + [2] * 40


def test_same_seed_gives_same_result(parameter_file_path, treatment_schedule):
    random_seed = 4343
    same_seeded_models = [
        SinotModel(parameter_file_path, random_seed) for _ in range(5)
    ]
    observed_outcome = [
        [model.observe_outcome(action, {}) for action in treatment_schedule]
        for model in same_seeded_models
    ]
    for left, right in itertools.pairwise(observed_outcome):
        assert left == right


def test_different_seed_gives_different_result(parameter_file_path, treatment_schedule):
    different_seeded_models = [SinotModel(parameter_file_path, i) for i in range(5)]
    observed_outcome = [
        [model.observe_outcome(action, {}) for action in treatment_schedule]
        for model in different_seeded_models
    ]
    for left, right in itertools.pairwise(observed_outcome):
        assert left != right
