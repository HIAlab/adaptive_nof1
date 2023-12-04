import numpy

import pandas
import scipy


## Code from: https://github.com/tonyduan/conjugate-bayes/blob/master/conjugate_bayes/models.py
class NIGLinearRegression(object):
    """
    The normal inverse-gamma prior for a linear regression model with unknown
    variance and unknown relationship. Specifically,
        1/σ² ~ Γ(a, b)
        β ~ N(0, σ²V)

    Parameters
    ----------
    mu: prior for N(mu, v) on the model β
    v:  prior for N(mu, v) on the model β
    a:  prior for Γ(a, b) on the inverse sigma2 of the distribution
    b:  prior for Γ(a, b) on the inverse sigma2 of the distribution
    """

    def __init__(self, mu, v, a, b):
        self.__dict__.update({"mu": mu, "v": v, "a": a, "b": b})

    def fit(self, x_tr, y_tr):
        m, _ = x_tr.shape
        mu_ast = numpy.linalg.inv(numpy.linalg.inv(self.v) + x_tr.T @ x_tr) @ (
            numpy.linalg.inv(self.v) @ self.mu + x_tr.T @ y_tr
        )
        v_ast = numpy.linalg.inv(numpy.linalg.inv(self.v) + x_tr.T @ x_tr)
        a_ast = self.a + 0.5 * m
        b_ast = self.b + 0.5 * (y_tr - x_tr @ self.mu).T @ numpy.linalg.inv(
            numpy.eye(m) + x_tr @ self.v @ x_tr.T
        ) @ (y_tr - x_tr @ self.mu.T)
        self.__dict__.update({"mu": mu_ast, "v": v_ast, "a": a_ast, "b": b_ast})

    def predict(self, x_te):
        scales = numpy.array([x.T @ self.v @ x for x in x_te]) + 1
        scales = (self.b / self.a * scales) ** 0.5
        return scipy.stats.t(df=2 * self.a, loc=x_te @ self.mu, scale=scales)

    def get_conditional_beta(self, sigma2):
        return scipy.stats.multivariate_normal(mean=self.mu, cov=sigma2 * self.v)

    def get_marginal_sigma2(self):
        return scipy.stats.invgamma(self.a, scale=self.b)


class BayesianLinearRegressionModel:
    def __init__(
        self,
        treatment_name="treatment",
        outcome_name="outcome",
        mean=0,
        v=0,
        alpha=0,
        beta=0,
    ):
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name

        self.mean = mean
        self.v = v
        self.alpha = alpha
        self.beta = beta

        self.df = None
        self.model = NIGLinearRegression(
            mu=self.mean, v=self.v, a=self.alpha, b=self.beta
        )

    def get_upper_confidence_bounds(self, variable_name, epsilon: float = 0.05):
        raise AssertionError("Not implemented")

    def update_posterior(self, history, number_of_treatments):
        self.history = history

    def __str__(self):
        return "BayesianLinearRegression"

    def series(self, intervention):
        return self.df[self.df[self.treatment_name] == intervention][self.outcome_name]

    def posterior(self, intervention):
        identity_matrix = numpy.eye(len(self.mean))
        return self.model.predict([identity_matrix[intervention]])

    def approximate_max_probabilities(self, number_of_treatments, context):
        assert len(self.mean) == number_of_treatments, "Wrong shape for mean"
        assert self.v.shape == (
            number_of_treatments,
            number_of_treatments,
        ), "Wrong shape for v"

        self.df = self.history.to_df()
        if len(self.df) == 0:
            return [0.5, 0.5]

        x = (
            pandas.get_dummies(
                pandas.Categorical(
                    self.df[self.treatment_name], categories=range(number_of_treatments)
                )
            )
            .to_numpy()
            .astype(float)
        )
        # print(x)
        # print(x.shape)
        self.model = NIGLinearRegression(
            mu=self.mean, v=self.v, a=self.alpha, b=self.beta
        )
        self.model.fit(x, self.df[self.outcome_name])
        # print(model.mu)

        sample_size = 1000
        identity_matrix = numpy.eye(number_of_treatments)
        samples = numpy.array(
            [
                self.model.predict([identity_matrix[intervention]]).rvs(
                    size=sample_size
                )
                for intervention in range(number_of_treatments)
            ]
        )
        max_indices = numpy.argmax(samples, axis=0)
        # print(samples)
        bin_counts = numpy.bincount(max_indices, minlength=number_of_treatments)
        return bin_counts / numpy.sum(bin_counts)

    def debug_data(self):
        if not self.model:
            return {
                "mean": self.mean,
                "v": self.v,
                "alpha": self.alpha,
                "beta": self.beta,
            }
        return {
            "mean": self.model.mu,
            "v": self.model.v,
            "alpha": self.model.a,
            "beta": self.model.b,
        }
