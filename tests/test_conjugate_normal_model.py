from adaptive_nof1.inference.conjugate_normal_model import ConjugateNormalModel
import pytest


from adaptive_nof1.helpers import array_almost_equal

from .helper_functions import outcomes_to_history


# TODO: Generate real test cases
@pytest.mark.sampling
@pytest.mark.parametrize(
    "treatments, priors, outcomes, expected",
    [
        ([0, 0, 1, 1], (2.5, 5, 1, 1), [3, 3, 3, 3], [0.5, 0.5]),
        ([0, 0, 1, 1], (2.5, 5, 1, 1), [3, 3, 5, 4], [0.28, 0.72]),
    ],
)
def test_conjugate_normal_model(treatments, priors, outcomes, expected):
    number_of_treatments = 2
    m, k, alpha, beta = priors
    model = ConjugateNormalModel(
        mean=m, l=k, alpha=alpha, beta=beta, seed=42, sample_size=5000
    )
    model.update_posterior(
        outcomes_to_history(treatments, outcomes), number_of_treatments
    )
    p = model.approximate_max_probabilities(number_of_treatments, {"t": len(outcomes)})
    assert array_almost_equal(p, expected, epsilon=0.02)
