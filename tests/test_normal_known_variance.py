from adaptive_nof1.inference.conjugate_normal_model import ConjugateNormalModel
import pytest


from adaptive_nof1.helpers import array_almost_equal
from adaptive_nof1.inference.normal_known_variance import NormalKnownVariance

from .helper_functions import outcomes_to_history


# TODO: Generate real test cases
@pytest.mark.sampling
@pytest.mark.parametrize(
    "treatments, outcomes, expected",
    [
        ([0, 0, 0, 0], [1, 1, -1, -1], [0.5, 0.5]),
        ([0, 0, 1, 1], [-1, 1, 2, -2], [0.50, 0.50]),
        ([0, 0, 1, 1], [-1, -2, -2, -3], [0.64, 0.36]),
    ],
)
def test_normal_known_variance_max_probabilities(treatments, outcomes, expected):
    number_of_treatments = 2
    model = NormalKnownVariance(
        prior_mean=0,
        prior_variance=1,
        variance=1,
    )
    model.update_posterior(
        outcomes_to_history(treatments, outcomes), number_of_treatments
    )
    p = model.approximate_max_probabilities(number_of_treatments, {"t": len(outcomes)})
    assert array_almost_equal(p, expected, epsilon=0.02)


@pytest.mark.parametrize(
    "treatments, outcomes, expected",
    [
        ([0, 0, 0, 0], [1, 1, -1, -1], [1.97, 3.3]),
        ([0, 0, 1, 1], [-1, 1, 2, -2], [2.2, 2.2]),
        ([0, 0, 1, 1], [-1, -2, -2, -3], [1.19, 0.52]),
    ],
)
def test_normal_known_variance_upper_confidence_bounds(treatments, outcomes, expected):
    number_of_treatments = 2
    model = NormalKnownVariance(
        prior_mean=0,
        prior_variance=1,
        variance=1,
    )
    model.update_posterior(
        outcomes_to_history(treatments, outcomes), number_of_treatments
    )
    p = model.get_upper_confidence_bounds(number_of_treatments)
    assert array_almost_equal(p, expected, epsilon=0.02)
