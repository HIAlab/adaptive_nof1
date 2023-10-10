from adaptive_nof1.helpers import series_to_indexed_array
import numpy as np
import pymc


# Inspired by the Thompson Sampling implementation in the Self-E app
class BetaModel:
    def __init__(self, treatment_name="treatment", outcome_name="outcome"):
        self.trace = None
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name

    def get_upper_confidence_bounds(self, variable_name, epsilon: float = 0.05):
        raise AssertionError("Not implemented")

    def dataframe_to_n_successes(self, df, number_of_treatments):
        mean_outcome = df[self.outcome_name].mean()
        return series_to_indexed_array(
            df.groupby(self.treatment_name, group_keys=True)[self.outcome_name].apply(
                lambda x: (x > mean_outcome).sum()
            ),
            min_length=number_of_treatments,
        )

    def dataframe_to_n_trials(self, df, number_of_treatments):
        return series_to_indexed_array(
            df.groupby(self.treatment_name, group_keys=True)[self.outcome_name].count(),
            min_length=number_of_treatments,
        )

    def update_posterior(self, history, number_of_treatments):
        df = history.to_df()

        if len(df) == 0:
            return

        n_successes = self.dataframe_to_n_successes(df, number_of_treatments)
        n_trials = self.dataframe_to_n_trials(df, number_of_treatments)
        n_failures = [b - a for a, b in zip(n_successes, n_trials)]

        assert len(n_successes) == number_of_treatments
        assert len(n_successes) == len(n_trials)
        assert len(n_successes) == len(n_failures)

        self.model = pymc.Model()
        with self.model:
            betas = pymc.Beta(
                "success_probabilities",
                alpha=[a + 1 for a in n_successes],
                beta=[a + 1 for a in n_failures],
                dims="type_number",
                shape=number_of_treatments,
            )
            self.trace = pymc.sample(100, progressbar=False)

    def __str__(self):
        return "BetaModel"

    def approximate_max_probabilities(self, number_of_treatments, context):
        if context["t"] == 0:
            return [1 / number_of_treatments] * number_of_treatments
        max_indices = np.ravel(
            self.trace["posterior"]["success_probabilities"].argmax(dim="type_number")
        )
        bin_counts = np.bincount(max_indices, minlength=number_of_treatments)
        return bin_counts / np.sum(bin_counts)
