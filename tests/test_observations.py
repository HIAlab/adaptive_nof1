import pytest

from adaptive_nof1.basic_types import *
from .fixtures import simple_history


def test_observation_equality():
    left = Observation(
        context={"activity": 10},
        treatment={"treatment": 0},
        outcome={"outcome": 2},
        t=0,
        patient_id=0,
    )
    right = Observation(
        context={"activity": 10},
        treatment={"treatment": 0},
        outcome={"outcome": 2},
        t=0,
        patient_id=0,
    )
    assert left == right


def test_observation_inequality():
    left = Observation(
        context={"activity": 10},
        treatment={"treatment": 0},
        outcome={"outcome": 2},
        t=0,
        patient_id=0,
    )

    right_different_context = Observation(
        context={"activity": 9},
        treatment={"treatment": 0},
        outcome={"outcome": 2},
        t=0,
        patient_id=0,
    )
    assert left != right_different_context

    right_different_treatment = Observation(
        context={"activity": 10},
        treatment={"treatment": 1},
        outcome={"outcome": 2},
        t=0,
        patient_id=0,
    )
    assert left != right_different_treatment

    right_different_outcome = Observation(
        context={"activity": 10},
        treatment={"treatment": 0},
        outcome={"outcome": 3},
        t=0,
        patient_id=0,
    )
    assert left != right_different_outcome


def test_observation_comparison():
    left = ({"outcome": 3},)
    right = ({"outcome": 2},)
    assert left == left
    assert left != right
