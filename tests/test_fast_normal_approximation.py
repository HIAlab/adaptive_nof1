from adaptive_nof1.inference.fast_normal_approximation import FastNormalApproximation
import pytest


from adaptive_nof1.helpers import array_almost_equal

from .helper_functions import outcomes_to_history


@pytest.mark.sampling
@pytest.mark.parametrize(
    "treatments, outcomes, expected",
    [
        ([0, 0, 1, 1], [2, 4, 3, 3], [0.5, 0.5]),
        ([0, 1], [3, 3], [0.5, 0.5]),
        ([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 5, 4, 3, 2, 1], [0.5, 0.5]),
        ([0], [1], [0.5, 0.5]),
        ([0, 0, 0, 1, 1, 1], [1, 2, 3, 4, 5, 6], [0.07, 0.93]),
    ],
)
def test_fast_normal_approximation_model(treatments, outcomes, expected):
    number_of_treatments = 2
    model = FastNormalApproximation()
    model.update_posterior(
        outcomes_to_history(treatments, outcomes), number_of_treatments
    )
    p = model.approximate_max_probabilities(number_of_treatments, {"t": len(outcomes)})
    assert array_almost_equal(p, expected, epsilon=0.02)
