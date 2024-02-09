from ctypes import c_void_p
import logging

import numpy as np
import pandas
import pymc
import arviz
from sklearn.metrics import classification_report
from typing import List


class BayesianModel:
    def __init__(self, treatment_name="treatment", outcome_name="outcome"):
        self.trace = None
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self._debug_data = {}

    def get_upper_confidence_bounds(self, variable_name, epsilon: float = 0.05):
        confidence_bounds = arviz.hdi(
            self.trace.posterior_predictive,
            var_names=[variable_name],
            hdi_prob=1 - epsilon,
        )
        # The data containts always (lower, upper), and we only want higher
        return confidence_bounds[variable_name][:, 1].to_numpy()

    def data_to_coefficient_matrix(self, df):
        data = df[self.coefficient_names].to_numpy()
        data.shape = (len(df), len(self.coefficient_names))
        return data

    def approximate_max_probabilities(self, number_of_treatments, _):
        max_indices = np.ravel(
            self.trace["posterior"]["average_treatment_effect"].argmax(
                dim="average_treatment_effect_dim_0"
            )
        )
        bin_counts = np.bincount(max_indices, minlength=number_of_treatments)
        return bin_counts / np.sum(bin_counts)

    def data_to_treatment_matrix(self, df, number_of_treatments):
        # Creating a Categorical Series makes get_dummies also create dummies for treatments which are not present in the dataset yet
        treatment_dummies = pandas.get_dummies(
            pandas.Categorical(
                df[self.treatment_name], categories=range(number_of_treatments)
            )
        )
        sorted_treatment_dummies = treatment_dummies.reindex(
            sorted(treatment_dummies.columns), axis=1
        )
        return pymc.floatX(sorted_treatment_dummies.to_numpy())

    @property
    def debug_data(self):
        return self._debug_data


class GaussianAverageTreatmentEffect(BayesianModel):
    def __str__(self):
        return "GaussianAverageTreatmentEffect"

    def update_posterior(self, history, number_of_treatments):
        # Simple model with baseline + treatment effects + noise
        df = history.to_df()

        self.model = pymc.Model()
        with self.model:
            logger = logging.getLogger("pymc")
            logger.disabled = False
            treatment_dummies = pandas.get_dummies(df[self.treatment_name])

            baseline = 0

            random_noise_variance = pymc.HalfNormal("random_noise_variance", sigma=100)
            average_treatment_effect = pymc.Normal(
                "average_treatment_effect", mu=0, sigma=100, shape=number_of_treatments
            )

            mu = baseline
            for treatment in treatment_dummies.columns:
                mu += (
                    np.array(treatment_dummies[treatment])
                    * average_treatment_effect[treatment]
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

        self.model = pymc.Model()
        with self.model:
            logger = logging.getLogger("pymc")
            logger.disabled = False
            treatment_dummies = pandas.get_dummies(df[self.treatment_name])

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
                    * average_treatment_effect[treatment]
                )

            outcome = pymc.Normal(
                "outcome",
                mu=mu,
                observed=df["outcome"],
                sigma=self.random_variance,
            )
            self.trace = pymc.sample(2000, progressbar=False)


class LinearAdditiveInferenceModel(BayesianModel):
    def __init__(self, coefficient_names, effect_variance, random_variance, **kwargs):
        self.coefficient_names = coefficient_names
        self.effect_variance = effect_variance
        self.random_variance = random_variance
        super().__init__(**kwargs)

    def __str__(self):
        return f"LinearAdditiveModel"

    def data_to_treatment_matrix(self, df, number_of_treatments):
        # Creating a Categorical Series makes get_dummies also create dummies for treatments which are not present in the dataset yet
        treatment_dummies = pandas.get_dummies(
            pandas.Categorical(
                df[self.treatment_name], categories=range(number_of_treatments)
            )
        )
        return pymc.floatX(treatment_dummies.to_numpy())

    def update_posterior(self, history, number_of_treatments):
        df = history.to_df()

        self.model = pymc.Model()
        with self.model:
            logger = logging.getLogger("pymc")
            logger.disabled = False

            treatment_selection_matrix = pymc.MutableData(
                "treatment_selection_matrix",
                self.data_to_treatment_matrix(df, number_of_treatments),
                dims=("obs_id", "treatment_id"),
            )
            coefficient_values = pymc.MutableData(
                "coefficient_values",
                self.data_to_coefficient_matrix(df),
                dims=("obs_id", "coefficient_id"),
            )  # n * number_of_coefficients
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
                shape=(number_of_treatments, len(self.coefficient_names)),
                dims=("treatment_id", "coefficient_id"),
            )

            for treatment_index in range(number_of_treatments):
                local_slopes = slopes[treatment_index]  # 1 * number_of_coefficients

                local_summand = pymc.math.dot(coefficient_values, local_slopes.T)
                masked_summand = pymc.floatX(
                    pymc.math.prod(
                        pymc.math.stack(
                            [
                                local_summand,
                                treatment_selection_matrix[:, treatment_index],
                            ]
                        ),
                        axis=0,
                    )
                )
                mu += masked_summand

            linear_regression = pymc.Deterministic(
                "linear_regression", var=mu, dims="obs_id"
            )

            outcome = pymc.Normal(
                "outcome",
                mu=linear_regression,
                sigma=self.random_variance,
                observed=df[self.outcome_name],
                dims="obs_id",
            )
            self.trace = pymc.sample(2000, progressbar=False)

    def approximate_max_probabilities(self, number_of_treatments, context):
        assert (
            self.trace is not None
        ), "You called `approximate_max_probabilites` without updating the posterior"

        df = pandas.DataFrame([context] * number_of_treatments)
        df[self.treatment_name] = range(number_of_treatments)

        with self.model:
            pymc.set_data(
                {
                    "coefficient_values": self.data_to_coefficient_matrix(df),
                    "treatment_selection_matrix": self.data_to_treatment_matrix(
                        df, number_of_treatments
                    ),
                }
            )  # n * number_of_coefficients
            pymc.sample_posterior_predictive(
                self.trace,
                var_names=["linear_regression"],
                extend_inferencedata=True,
            )

        max_indices = arviz.extract(
            self.trace.posterior_predictive
        ).linear_regression.argmax(dim="obs_id")
        bin_counts = np.bincount(max_indices, minlength=number_of_treatments)
        return bin_counts / np.sum(bin_counts)


class BernoulliLogItInferenceModel(BayesianModel):
    def __init__(self, coefficient_names, effect_variance, random_variance, **kwargs):
        self.coefficient_names = coefficient_names
        self.effect_variance = effect_variance
        self.random_variance = random_variance
        super().__init__(**kwargs)

    def __str__(self):
        return f"BernoulliLogItInferenceModel"

    def update_posterior(self, history, number_of_treatments):
        df = history.to_df()

        self.model = pymc.Model()
        with self.model:
            logger = logging.getLogger("pymc")
            logger.disabled = False

            treatment_selection_matrix = pymc.MutableData(
                "treatment_selection_matrix",
                self.data_to_treatment_matrix(df, number_of_treatments),
                dims=("obs_id", "treatment_id"),
            )
            coefficient_values = pymc.MutableData(
                "coefficient_values",
                self.data_to_coefficient_matrix(df),
                dims=("obs_id", "coefficient_id"),
            )  # n * number_of_coefficients
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
                shape=(number_of_treatments, len(self.coefficient_names)),
                dims=("treatment_id", "coefficient_id"),
            )

            for treatment_index in range(number_of_treatments):
                local_slopes = slopes[treatment_index]  # 1 * number_of_coefficients

                local_summand = pymc.math.dot(coefficient_values, local_slopes.T)
                masked_summand = pymc.floatX(
                    pymc.math.prod(
                        pymc.math.stack(
                            [
                                local_summand,
                                treatment_selection_matrix[:, treatment_index],
                            ]
                        ),
                        axis=0,
                    )
                )
                mu += masked_summand

            linear_regression = pymc.Deterministic(
                "linear_regression", var=mu, dims="obs_id"
            )

            # Transfrom from [-1, 1] to [0, 1]
            normalized_observed = (df[self.outcome_name] + 1) / 2

            # transformation
            linear_regression_transformed = pymc.Deterministic(
                "linear_regression_transformed",
                var=pymc.math.sigmoid(linear_regression),
                dims="obs_id",
            )

            outcome = pymc.Bernoulli(
                "outcome",
                p=linear_regression_transformed,
                observed=normalized_observed,
                dims="obs_id",
            )
            self.trace = pymc.sample(2000, progressbar=False)

    def predict_for_history(self, history, number_of_treatments):
        df = history.to_df()
        with self.model:
            pymc.set_data(
                {
                    "coefficient_values": self.data_to_coefficient_matrix(df),
                    "treatment_selection_matrix": self.data_to_treatment_matrix(
                        df, number_of_treatments
                    ),
                }
            )
            return pymc.sample_posterior_predictive(
                self.trace,
                var_names=["outcome"],
            )

    def classification_scores(self, history, number_of_treatments):
        trace = self.predict_for_history(history, number_of_treatments)
        predicted_outcome = arviz.extract(
            trace.posterior_predictive, var_names="outcome", num_samples=1
        )
        outcome = (history.to_df()[self.outcome_name] + 1) / 2
        return classification_report(y_true=outcome, y_pred=predicted_outcome)

    def approximate_max_probabilities(self, number_of_treatments, context):
        assert (
            self.trace is not None
        ), "You called `approximate_max_probabilites` without updating the posterior"

        df = pandas.DataFrame([context] * number_of_treatments)
        df[self.treatment_name] = range(number_of_treatments)

        with self.model:
            pymc.set_data(
                {
                    "coefficient_values": self.data_to_coefficient_matrix(df),
                    "treatment_selection_matrix": self.data_to_treatment_matrix(
                        df, number_of_treatments
                    ),
                }
            )  # n * number_of_coefficients
            pymc.sample_posterior_predictive(
                self.trace,
                var_names=["linear_regression_transformed"],
                extend_inferencedata=True,
            )

        max_indices = arviz.extract(
            self.trace.posterior_predictive
        ).linear_regression_transformed.argmax(dim="obs_id")
        bin_counts = np.bincount(max_indices, minlength=number_of_treatments)
        return bin_counts / np.sum(bin_counts)
