import pytest

from adaptive_nof1.basic_types import *


@pytest.fixture
def simple_history():
    return History(
        observations=[
            Observation(
                context={"activity": 10},
                treatment={"treatment": 0},
                outcome={"outcome": 2},
                t=0,
                patient_id=0,
            ),
            Observation(
                context={"activity": 20},
                treatment={"treatment": 1},
                outcome={"outcome": 3},
                t=0,
                patient_id=0,
            ),
            Observation(
                context={"activity": 30},
                treatment={"treatment": 2},
                outcome={"outcome": 3},
                t=0,
                patient_id=0,
            ),
        ]
    )


def test_history_to_df(simple_history):
    expected_df = pd.DataFrame(
        {"treatment": [0, 1, 2], "activity": [10, 20, 30], "outcome": [2, 3, 3]}
    )
    assert expected_df.reindex(sorted(expected_df.columns), axis=1).equals(
        simple_history.to_df().reindex(sorted(expected_df.columns), axis=1)
    )


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
