from adaptive_nof1.helpers import series_to_indexed_array
import numpy

from scipy.stats import invgamma, norm


class ConjugateNormalModel:
    def __init__(
        self,
        treatment_name="treatment",
        outcome_name="outcome",
        mean=0,
        l=0,
        alpha=0,
        beta=0,
        seed=None
    ):
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self.rng = numpy.random.default_rng(seed)
        invgamma.random_state = self.rng
        norm.random_state = self.rng

        self.mean = mean
        self.l = l
        self.alpha = alpha
        self.beta = beta

        self.df = None

    def get_upper_confidence_bounds(self, variable_name, epsilon: float = 0.05):
        raise AssertionError("Not implemented")

    def update_posterior(self, history, number_of_treatments):
        self.history = history

    def __str__(self):
        return "ConjugateNormalModel"

    def n(self, intervention):
        return len(self.series(intervention))

    def sample_mean(self, intervention):
        if self.n(intervention) == 0:
            return 0.0
        return self.series(intervention).mean()

    def series(self, intervention):
        return self.df[self.df[self.treatment_name] == intervention][self.outcome_name]

    def var(self, intervention):
        var = numpy.var(self.series(intervention))
        if numpy.isnan(var):
            return 0.0
        return var

    def mean_update(self, intervention):
        return (
            self.l * self.mean + self.n(intervention) * self.sample_mean(intervention)
        ) / (self.l + self.n(intervention))

    def l_update(self, intervention):
        return self.l + self.n(intervention)

    def alpha_update(self, intervention):
        return self.alpha + self.n(intervention) / 2

    def beta_update(self, intervention):
        return (
            self.beta
            + 0.5 * self.var(intervention)
            + self.n(intervention)
            * self.l
            * (self.sample_mean(intervention) - self.mean) ** 2
            / (self.l + self.n(intervention))
            * 2
        )

    def sample_normal_inverse_gamma(self, mean, l, alpha, beta, sample_size, number_of_treatments):
        # Sample from our updated distributions
        sample_size = 1000
        invgamma.random_state = self.rng
        sigma_squared_samples = invgamma.rvs(a=alpha, scale=beta, size=(sample_size, number_of_treatments))
        samples = norm.rvs(
            loc=mean, scale=numpy.sqrt(sigma_squared_samples / l), size=(sample_size, number_of_treatments)
        )
        return samples

    def approximate_max_probabilities(self, number_of_treatments, context):
        self.df = self.history.to_df()

        # calculate posterior parameters:
        # See https://en.wikipedia.org/wiki/Conjugate_prior and then Normal with unkown mean and variance
        mean = []
        l = []
        alpha = []
        beta = []

        for intervention in range(number_of_treatments):
            mean.append(self.mean_update(intervention))
            l.append(self.l_update(intervention))
            alpha.append(self.alpha_update(intervention))
            beta.append(self.beta_update(intervention))

        sample_size = 1000
        samples = self.sample_normal_inverse_gamma(mean, l, alpha, beta, sample_size, number_of_treatments)

        max_indices = numpy.argmax(samples, axis=1)

        bin_counts = numpy.bincount(max_indices, minlength=number_of_treatments)
        return bin_counts / numpy.sum(bin_counts)
