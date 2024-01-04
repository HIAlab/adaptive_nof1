from adaptive_nof1.helpers import series_to_indexed_array
import numpy

from scipy.stats import invgamma, norm
import scipy


class ConstantProbability:
    def __init__(
        self,
        probabilities,
        treatment_name="treatment",
        outcome_name="outcome",
    ):
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self.probabilities = probabilities
        self._debug_data = {}

    def additional_config(self):
        return {}

    def get_upper_confidence_bounds(self, variable_name, epsilon: float = 0.05):
        raise AssertionError("Not implemented")

    def update_posterior(self, history, number_of_treatments):
        pass

    def __str__(self):
        return f"ConstantProbability:{self.probabilities}"

    def approximate_max_probabilities(self, number_of_treatments, context):
        return self.probabilities

    def debug_data(self):
        return self._debug_data
