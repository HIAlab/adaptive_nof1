import pytest

from adaptive_nof1.policies import SequentialHalving

from tests.mocks import MockBayesianModel

from tests.helper_functions import outcomes_to_history

from unittest.mock import Mock


@pytest.mark.parametrize(
    "number_of_interventions, length, expected",
    [
        (4, 16, [0, 1, 2, 3] * 2 + [2, 3] * 4),
        (2, 16, [0, 1] * 8),
        (8, 8 * 3, list(range(8)) + [4, 5, 6, 7] * 2 + [6, 7] * 4),
    ],
)
def test_sequential_halving(number_of_interventions, length, expected):
    probabilities = list(range(number_of_interventions))

    sequential_halving = SequentialHalving(
        inference_model=MockBayesianModel(max_probabilities=probabilities),
        length=length,
    )
    sequential_halving.number_of_interventions = number_of_interventions

    # Mock Rng
    sequential_halving.rng = Mock()
    sequential_halving.rng.permutation = lambda x: sorted(x)

    mock_outcomes = [0] * length

    list_of_actions = []
    for t in range(length):
        list_of_actions.append(
            sequential_halving.choose_action(
                outcomes_to_history(
                    list_of_actions, mock_outcomes[0 : len(list_of_actions)]
                ),
                {"t": t},
            )["treatment"]
        )

    assert list_of_actions == expected
