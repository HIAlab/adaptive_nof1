from adaptive_nof1.basic_types import History, Observation
from .fixtures import simple_history
import pandas


def test_history_to_df(simple_history):
    expected_df = pandas.DataFrame(
        {"treatment": [0, 1, 2], "activity": [10, 20, 30], "outcome": [2, 3, 3]}
    )
    assert expected_df.reindex(sorted(expected_df.columns), axis=1).equals(
        simple_history.to_df().reindex(sorted(expected_df.columns), axis=1)
    )
