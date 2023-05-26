import logging

import arviz as az
import numpy as np
import pandas as pd
import pymc
import pytensor


class BayesianModel:
    def __init__(self):
        self.trace = None

    def get_upper_confidence_bounds(self, variable_name, epsilon: float = 0.05):
        return az.hdi(
            self.trace.posterior, var_names=[variable_name], hdi_prob=1 - epsilon
        )

    def approximate_max_probabilities(self, number_of_treatments):
        max_indices = np.ravel(
            self.trace["posterior"]["average_treatment_effect"].argmax(
                dim="average_treatment_effect_dim_0"
            )
        )
        bin_counts = np.bincount(max_indices, minlength=number_of_treatments)
        return bin_counts / np.sum(bin_counts)


class GaussianAverageTreatmentEffect(BayesianModel):
    def __str__(self):
        return "GaussianAverageTreatmentEffect"

    def update_posterior(self, history, number_of_treatments):
        # Simple model with baseline + treatment effects + noise
        df = history.to_df()

        model = pymc.Model()
        with model:
            logger = logging.getLogger("pymc")
            logger.disabled = False
            treatment_dummies = pd.get_dummies(df["treatment"])

            baseline = 0

            random_noise_variance = pymc.HalfNormal("random_noise_variance", sigma=100)
            average_treatment_effect = pymc.Normal(
                "average_treatment_effect", mu=0, sigma=100, shape=number_of_treatments
            )

            mu = baseline
            for treatment in treatment_dummies.columns:
                mu += (
                    np.array(treatment_dummies[treatment])
                    * average_treatment_effect[treatment - 1]
                )

            outcome = pymc.Normal(
                "outcome", mu=mu, observed=df["outcome"], sigma=random_noise_variance
            )
            self.trace = pymc.sample(2000, progressbar=False)


class FixedVarianceNormalEffect(BayesianModel):
    def __init__(self, effect_variance, random_variance):
        self.effect_variance = effect_variance
        self.random_variance = random_variance

    def __str__(self):
        return "FixedVarianceNormalEffect"

    def update_posterior(self, history, number_of_treatments):
        # Simple model with baseline + treatment effects + noise
        df = history.to_df()

        model = pymc.Model()
        with model:
            logger = logging.getLogger("pymc")
            logger.disabled = False
            treatment_dummies = pd.get_dummies(df["treatment"])

            average_treatment_effect = pymc.Normal(
                "average_treatment_effect",
                mu=0,
                sigma=self.effect_variance,
                shape=number_of_treatments,
            )

            mu = 0
            for treatment in treatment_dummies.columns:
                mu += (
                    np.array(treatment_dummies[treatment])
                    * average_treatment_effect[treatment - 1]
                )

            outcome = pymc.Normal(
                "outcome",
                mu=mu,
                observed=df["outcome"],
                sigma=self.random_variance,
            )
            self.trace = pymc.sample(2000, progressbar=False)


class LinearAdditiveModel(BayesianModel):
    def __init__(self, number_of_coefficients, effect_variance, random_variance):
        self.number_of_coefficients = number_of_coefficients
        self.effect_variance = effect_variance
        self.random_variance = random_variance

    def __str__(self):
        return f"LinearAdditiveModel"

    def update_posterior(self, history, number_of_treatments):
        df = history.to_df()

        model = pymc.Model()
        with model:
            logger = logging.getLogger("pymc")
            logger.disabled = False
            treatment_dummies = pd.get_dummies(df["treatment"])

            treatment_selection_matrix = treatment_dummies.to_numpy()
            intercept = pymc.Normal(
                "intercept",
                mu=0,
                sigma=self.effect_variance,
                shape=number_of_treatments,
            )
            mu = pymc.math.dot(treatment_selection_matrix, intercept.T)
            slopes = pymc.Normal(
                "slopes",
                mu=0,
                sigma=self.effect_variance,
                shape=(number_of_treatments, self.number_of_coefficients),
            )

            for treatment in treatment_dummies.columns:
                local_coefficients = df[
                    range(self.number_of_coefficients)
                ].to_numpy()  # n * number_of_coefficien
                local_slopes = slopes[treatment - 1]  # 1 * number_of_coefficients

                local_summand = pymc.math.dot(local_coefficients, local_slopes.T)
                masked_summand = pymc.floatX(
                    pymc.math.prod(
                        pymc.math.stack(
                            [
                                local_summand,
                                treatment_selection_matrix[:, treatment - 1],
                            ]
                        ),
                        axis=0,
                    )
                )
                mu += masked_summand

            outcome = pymc.Normal(
                "outcome",
                mu=mu,
                observed=df["outcome"],
                sigma=self.random_variance,
            )
            self.trace = pymc.sample(2000, progressbar=False)

    # TODO
    def approximate_max_probabilities(self, number_of_treatments):
        max_indices = np.ravel(
            self.trace["posterior"]["intercept"].argmax(dim="intercept_dim_0")
        )
        bin_counts = np.bincount(max_indices, minlength=number_of_treatments)
        return bin_counts / np.sum(bin_counts)
