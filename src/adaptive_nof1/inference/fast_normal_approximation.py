from adaptive_nof1.helpers import series_to_indexed_array
import numpy as np
import pymc

import scipy.stats as stats


class FastNormalApproximation:
    def __init__(self, treatment_name="treatment", outcome_name="outcome"):
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name

    def get_upper_confidence_bounds(self, variable_name, epsilon: float = 0.05):
        raise AssertionError("Not implemented")

    def update_posterior(self, history, number_of_treatments):
        self.history = history

    def __str__(self):
        return "FastNormalApproximation"

    def approximate_max_probabilities(self, number_of_treatments, context):
        if number_of_treatments != 2:
            raise AssertionError("Not implemented")

        df = self.history.to_df()

        if len(df) == 0:
            return [0.5, 0.5]

        # Approximate mean and variance per treatment
        groupby = df.groupby(self.treatment_name, group_keys=True)
        means = series_to_indexed_array(groupby[self.outcome_name].mean())
        stds = series_to_indexed_array(groupby[self.outcome_name].std())

        if len(means) != 2:
            return [0.5, 0.5]

        # See: https://stats.stackexchange.com/questions/50501/probability-of-one-random-variable-being-greater-than-another
        # Let X, Y be the Normal distributions
        # Pr(X > Y) == Pr(X - Y > 0)
        # X - Y == N(m(X) - m(Y), V(X) + V(Y))
        mean = means[0] - means[1]
        std = stds[0] + stds[1]
        p = 1 - stats.norm.cdf(0, mean, std)
        return [p, 1 - p]
