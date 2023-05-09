import pymc
import numpy as np
import pandas as pd
import arviz as az

import logging


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

            random_noise_variance = pymc.HalfNormal("random_noise_variance", sigma=10)
            average_treatment_effect = pymc.Normal(
                "average_treatment_effect", mu=0, sigma=100, shape=number_of_treatments
            )

            mu = baseline
            for treatment in treatment_dummies.columns:
                treatment_index = treatment - 1
                mu += (
                    np.array(treatment_dummies[treatment])
                    * average_treatment_effect[treatment_index]
                )

            outcome = pymc.Normal(
                "outcome", mu=mu, observed=df["outcome"], sigma=random_noise_variance
            )
            self.trace = pymc.sample(4000, progressbar=False)
