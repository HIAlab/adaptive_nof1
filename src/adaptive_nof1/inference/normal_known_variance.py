from numpy.core.fromnumeric import mean
from adaptive_nof1.helpers import series_to_indexed_array
import numpy

from scipy.stats import invgamma, norm
import scipy
import torch


# See wikipedia.org/wiki/Conjugate_prior
# Normal with known variance \sigma^2 for the formulas
class NormalKnownVariance:
    def __init__(
        self,
        treatment_name="treatment",
        outcome_name="outcome",
        prior_mean=0,
        prior_variance=1,
        variance=1,
        seed=None,
    ):
        assert variance > 0, "Variance must be positive"
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self.rng = numpy.random.default_rng(seed)
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.variance = variance
        self.number_of_interventions = None

        self.df = None
        self._debug_data = {"mean": prior_mean, "variance": prior_variance}

    # This is an upper bound for the mean effect.
    # One could also upper bound the reward, which would add additional variance from self.variance.
    # For selection in UCB, this is irrelevant, since the max is chosen anyway.
    def get_upper_confidence_bounds(self, variable_name, epsilon: float = 0.05):
        assert variable_name == self.outcome_name, "Only outcome variable supported"
        assert (
            self.number_of_interventions is not None
        ), "Do not call get_upper_confidence_bounds without previously calling update_posterior"
        mean, variance = self.posterior_parameters(self.number_of_interventions)
        upper_confidence_bounds = norm.ppf(
            1 - epsilon,
            loc=mean,
            scale=numpy.sqrt(numpy.array(variance)),
        )
        return upper_confidence_bounds

    def update_posterior(self, history, number_of_treatments):
        self.number_of_interventions = number_of_treatments
        self.history = history
        self.df = self.history.to_df()
        mean, variance = self.posterior_parameters(number_of_treatments)
        self._debug_data = {"mean": mean, "variance": variance}

    def __str__(self):
        return f"NormalKnownVariance({self.prior_mean}, {self.prior_variance}, {self.variance})"

    def n(self, intervention):
        return len(self.series(intervention))

    def series(self, intervention):
        return self.df[self.df[self.treatment_name] == intervention][self.outcome_name]

    def var(self, intervention):
        var = numpy.var(self.series(intervention))
        return var

    def sum(self, intervention):
        return numpy.sum(self.series(intervention))

    def mean_update(self, intervention):
        return self.variance_update(intervention) * (
            self.prior_mean / self.prior_variance
            + (self.sum(intervention) / self.variance)
        )

    def variance_update(self, intervention):
        return 1.0 / (
            (1.0 / self.prior_variance) + self.n(intervention) / self.variance
        )

    def sample_posterior_predictive(
        self, mean, variance, sample_size, number_of_treatments
    ):
        # Sample from our updated distributions
        samples = norm.rvs(
            loc=mean,
            scale=numpy.sqrt(numpy.array(variance) + self.variance),
            size=(sample_size, number_of_treatments),
        )
        return samples

    def multivariate_normal_distribution(self):
        mean = self._debug_data["mean"]
        variance = self._debug_data["variance"]
        cov = torch.diag_embed(torch.tensor(numpy.sqrt(variance)))
        return torch.distributions.MultivariateNormal(torch.tensor(mean), cov)

    def posterior_parameters(self, number_of_treatments):
        mean = []
        variance = []

        for intervention in range(number_of_treatments):
            mean.append(self.mean_update(intervention))
            variance.append(self.variance_update(intervention))

        return mean, variance

    def sample_posterior_means(self, mean, variance, sample_size, number_of_treatments):
        samples = norm.rvs(
            loc=mean,
            scale=numpy.sqrt(numpy.array(variance)),
            size=(sample_size, number_of_treatments),
        )
        return samples

    # This calculates the probability from our model that each treatment is the best by sampling from the mean values
    def approximate_max_probabilities(self, number_of_treatments, context):
        # calculate posterior parameters:
        # See https://en.wikipedia.org/wiki/Conjugate_prior and then Normal with known variance

        mean, variance = self.posterior_parameters(number_of_treatments)
        self._debug_data = {"mean": mean, "variance": variance}

        sample_size = 5000
        samples = self.sample_posterior_means(
            mean, variance, sample_size, number_of_treatments
        )

        max_indices = numpy.argmax(samples, axis=1)

        bin_counts = numpy.bincount(max_indices, minlength=number_of_treatments)
        return bin_counts / numpy.sum(bin_counts)

    @property
    def debug_data(self):
        return self._debug_data
