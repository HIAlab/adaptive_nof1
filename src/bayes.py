import arviz as az
import logging
import numpy as np
import pandas as pd
import pymc


class BayesianInference:
    def get_upper_confidence_bounds(self, variable_name, epsilon: float = 0.05):
        return az.hdi(
            self.trace.posterior, var_names=[variable_name], hdi_prob=1 - epsilon
        )

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

    def approximate_max_probabilities(self):
        max_indices = np.ravel(
            self.trace["posterior"]["average_treatment_effect"].argmax(
                dim="average_treatment_effect_dim_0"
            )
        )
        bin_counts = np.bincount(max_indices)
        return bin_counts / np.sum(bin_counts)
