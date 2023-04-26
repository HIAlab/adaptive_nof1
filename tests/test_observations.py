from src.observation import History, Observation, Context, Treatment, Outcome
import pytest

import pandas as pd


@pytest.fixture
def simple_history():
    return History(
        observations=[
            Observation(
                context={"activity": 10},
                treatment=Treatment(**{"i": 0}),
                outcome=Outcome(**{"primary_outcome": 2}),
            ),
            Observation(
                context={"activity": 20},
                treatment=Treatment(**{"i": 1}),
                outcome=Outcome(**{"primary_outcome": 3}),
            ),
            Observation(
                context={"activity": 30},
                treatment=Treatment(**{"i": 2}),
                outcome=Outcome(**{"primary_outcome": 3}),
            ),
        ]
    )


def test_history_to_df(simple_history):
    expected_df = pd.DataFrame(
        {"treatment": [0, 1, 2], "activity": [10, 20, 30], "primary_outcome": [2, 3, 3]}
    )
    assert expected_df.reindex(sorted(expected_df.columns), axis=1).equals(
        simple_history.to_df().reindex(sorted(expected_df.columns), axis=1)
    )
