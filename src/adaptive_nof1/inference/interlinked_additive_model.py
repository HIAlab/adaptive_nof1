from .bayes import BayesianModel
from typing import List

from adaptive_nof1.helpers import index_from_subset, index_to_actions, index_to_values

import pymc
import pandas
import numpy
import arviz


def flatten(arrays):
    return [element for array in arrays for element in array]


class InterlinkedAdditiveModel(BayesianModel):
    def __init__(
        self,
        action_dimensions: List[int],
        action_names: List[str],
        coefficient_names_per_treatment: List[List[str]],
        **kwargs,
    ):
        self.action_dimensions = action_dimensions
        self.action_names = action_names
        self.coefficient_names_per_treatment = coefficient_names_per_treatment
        self.coefficient_names = list(
            numpy.unique(flatten(coefficient_names_per_treatment))
        )
        self.model = None
        super().__init__(**kwargs)

    def __str__(self):
        return "InterlinkedAdditiveModel"

    def data_to_treatment_indices(self, df):
        return pymc.intX((df[self.action_names] - 1).to_numpy())

    def setup_model(self):
        empty_df = pandas.DataFrame(
            columns=list(
                numpy.unique(
                    self.action_names + self.coefficient_names + [self.outcome_name]
                )
            ),
            dtype=float,
        )
        self.model = pymc.Model()
        with self.model:
            treatment_indices = pymc.MutableData(
                "treatment_indices",
                self.data_to_treatment_indices(empty_df),
                dims=("obs_id", "treatment_number"),
            )
            observed_outcomes = pymc.MutableData(
                "observed_outcomes", empty_df[self.outcome_name], dims="obs_id"
            )

            mu = 0
            for treatment_number in range(len(self.action_dimensions)):
                intercept = pymc.Normal(
                    f"intercept_{self.action_names[treatment_number]}",
                    mu=0,
                    sigma=1,
                    shape=self.action_dimensions[treatment_number],
                    dims=f"treatment_{self.action_names[treatment_number]}",
                )
                intercept_summand = intercept[treatment_indices[:, treatment_number]]
                mu += intercept_summand

                coefficient_names_for_treatment = self.coefficient_names_per_treatment[
                    treatment_number
                ]
                if len(coefficient_names_for_treatment) > 0:
                    # For an extensive example with explaination see ``test_pytensor_pymc.py::test_model``
                    slopes = pymc.Normal(
                        f"slopes_{self.action_names[treatment_number]}",
                        mu=0,
                        sigma=1,
                        shape=(
                            self.action_dimensions[treatment_number],
                            len(self.coefficient_names_per_treatment[treatment_number]),
                        ),
                        dims=(
                            f"treatment_number_{self.action_names[treatment_number]}",
                            f"coefficient_number_{self.action_names[treatment_number]}",
                        ),
                    )
                    coefficients_for_treatment = pymc.MutableData(
                        f"coefficients_for_treatment_{self.action_names[treatment_number]}",
                        self.coefficients_for_treatment(treatment_number, empty_df),
                        dims=(
                            "obs_id",
                            f"coefficient_number_{self.action_names[treatment_number]}",
                        ),
                    )
                    slopes_for_applied_treatments = slopes[
                        treatment_indices[:, treatment_number]
                    ]
                    coefficient_summand = pymc.math.extract_diag(
                        pymc.math.dot(
                            coefficients_for_treatment, slopes_for_applied_treatments.T
                        )
                    )
                    mu += coefficient_summand

                # print(f"intercept[treatment_indices[:, treatment_number]]{intercept[treatment_indices[:, treatment_number]].eval()}")
                # print(f"(coefficient_values[:, coefficient_indices] * slopes.T)[:, treatment_indices[:, treatment_number]]{(coefficient_values[:, coefficient_indices] * slopes.T)[:, treatment_indices[:, treatment_number]][:, 0].eval()}")
                # print(f"mu:{mu.eval()}")

            outcome = pymc.Normal(
                "outcome",
                mu=mu,
                sigma=1,
                observed=observed_outcomes,
                dims="obs_id",
            )

    def coefficients_for_treatment(self, treatment_number, df):
        coefficient_values = self.data_to_coefficient_matrix(df)
        coefficient_names_for_treatment = self.coefficient_names_per_treatment[
            treatment_number
        ]
        coefficient_indices = index_from_subset(
            self.coefficient_names,
            coefficient_names_for_treatment,
        )
        coefficients_for_treatment = coefficient_values[:, coefficient_indices]
        return coefficients_for_treatment

    def update_posterior(self, history, _):
        df = history.to_df()

        if not self.model:
            self.setup_model()

        coefficient_values = {
            f"coefficients_for_treatment_{self.action_names[treatment_number]}": self.coefficients_for_treatment(
                treatment_number, df
            )
            for treatment_number in range(len(self.action_names))
        }

        with self.model:
            pymc.set_data(
                {
                    "treatment_indices": self.data_to_treatment_indices(df),
                    "observed_outcomes": df[self.outcome_name],
                    **coefficient_values,
                }
            )
            self.trace = pymc.sample(2000, progressbar=False)

    def approximate_max_probabilities(self, number_of_treatments, context):
        assert (
            self.trace is not None
        ), "You called `approximate_max_probabilites` without updating the posterior"

        df = pandas.DataFrame(
            [context] * number_of_treatments,
            columns=self.action_names + self.coefficient_names + [self.outcome_name],
        )
        df["treatment_index"] = range(number_of_treatments)
        actions = [
            index_to_actions(treatment_index, self.action_dimensions, self.action_names)
            for treatment_index in range(number_of_treatments)
        ]
        for name in self.action_names:
            df[name] = [action[name] for action in actions]

        # Eliminate duplicate columns
        df = df.loc[:, ~df.columns.duplicated()].copy()

        coefficient_values = {
            f"coefficients_for_treatment_{self.action_names[treatment_number]}": self.coefficients_for_treatment(
                treatment_number, df
            )
            for treatment_number in range(len(self.action_names))
        }

        with self.model:
            pymc.set_data(
                {
                    "treatment_indices": self.data_to_treatment_indices(df),
                    **coefficient_values,
                },
            )  # n * number_of_coefficients
            pymc.sample_posterior_predictive(
                self.trace,
                var_names=["outcome"],
                extend_inferencedata=True,
            )

        max_indices = arviz.extract(self.trace.posterior_predictive).outcome.argmax(
            dim="obs_id"
        )
        bin_counts = numpy.bincount(max_indices, minlength=number_of_treatments)
        return bin_counts / numpy.sum(bin_counts)
