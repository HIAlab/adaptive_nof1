from adaptive_nof1.helpers import series_to_indexed_array
import numpy

from scipy.stats import invgamma, norm
import scipy


# See wikipedia.org/wiki/Conjugate_prior
# Normal with known variance \sigma^2 for the formulas
class NormalKnownVariance:
    def __init__(
        self,
        treatment_name="treatment",
        outcome_name="outcome",
        mean=0,
        variance=0,
        seed=None,
    ):
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self.rng = numpy.random.default_rng(seed)
        self.mean = mean
        self.variance = variance

        self.df = None
        self._debug_data = {"mean": mean, "variance": variance}

    def get_upper_confidence_bounds(self, variable_name, epsilon: float = 0.05):
        raise AssertionError("Not implemented")

    def update_posterior(self, history, number_of_treatments):
        self.history = history

    def __str__(self):
        return f"NormalKnownVariance({self.mean}, {self.variance})"

    def n(self, intervention):
        return len(self.series(intervention))

    def series(self, intervention):
        return self.df[self.df[self.treatment_name] == intervention][self.outcome_name]

    def var(self, intervention):
        var = numpy.var(self.series(intervention))
        if numpy.isnan(var):
            return 0.0
        return var

    def sum(self, intervention):
        return numpy.sum(self.series(intervention))

    def mean_update(self, intervention):
        if self.var(intervention) == 0:
            return self.mean
        return self.variance_update(intervention) * (
            self.mean / self.variance + self.sum(intervention) / self.var(intervention)
        )

    def variance_update(self, intervention):
        if self.var(intervention) == 0:
            return self.variance
        return 1 / ((1 / self.variance) + self.n(intervention) / self.var(intervention))

    def sample_posterior_predictive(
        self, mean, variance, sample_size, number_of_treatments
    ):
        # Sample from our updated distributions
        sample_size = 10000
        invgamma.random_state = self.rng
        samples = norm.rvs(
            loc=mean,
            scale=numpy.array(variance) + self.variance,
            size=(sample_size, number_of_treatments),
        )
        return samples

    def approximate_max_probabilities(self, number_of_treatments, context):
        self.df = self.history.to_df()

        # calculate posterior parameters:
        # See https://en.wikipedia.org/wiki/Conjugate_prior and then Normal with known variance
        mean = []
        variance = []

        for intervention in range(number_of_treatments):
            mean.append(self.mean_update(intervention))
            variance.append(self.variance_update(intervention))

        self._debug_data = {"mean": mean, "variance": variance}

        sample_size = 1000
        samples = self.sample_posterior_predictive(
            mean, variance, sample_size, number_of_treatments
        )

        max_indices = numpy.argmax(samples, axis=1)

        bin_counts = numpy.bincount(max_indices, minlength=number_of_treatments)
        return bin_counts / numpy.sum(bin_counts)

    def debug_data(self):
        return self._debug_data