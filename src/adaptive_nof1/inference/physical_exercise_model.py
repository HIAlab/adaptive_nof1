import numpy
import pandas
import arviz
import pymc

from adaptive_nof1.inference.bayes import BayesianModel

class PhysicalExerciseModel(BayesianModel):
    def __init__(
        self,
        dimension_for_type,
        possible_actions,
        **kwargs,
    ):
        self.possible_actions = possible_actions
        self.dimension_for_type = dimension_for_type
        self.action_name = "activity_index"
        self.coefficient_names = [
            "type",
            "intensity",
            "duration",
            "current_pain",
            "mean_duration",
            "mean_intensity",
        ]
        self.model = None
        super().__init__(**kwargs)

    def __str__(self):
        return "PhysicalExerciseModel"

    def data_to_treatment_indices(self, df):
        return pymc.intX((df[self.action_name]).to_numpy())

    def setup_model(self):
        empty_df = pandas.DataFrame(
            columns=list(
                numpy.unique(
                    [self.action_name] + self.coefficient_names + ["pain_reduction"]
                )
            ),
            dtype=float,
        )
        self.model = pymc.Model()
        with self.model:
            mu = 0
            type_intercept = pymc.Normal(
                "type_intercept",
                mu=0,
                sigma=1,
                dims="type_number",
                shape=self.dimension_for_type,
            )
            types = pymc.MutableData(
                "types", pymc.intX(empty_df["type"]), dims="obs_id"
            )
            intensities = pymc.MutableData(
                "intensities", pymc.floatX(empty_df["intensity"]), dims="obs_id"
            )
            mean_intensities = pymc.MutableData(
                "mean_intensities", pymc.floatX(empty_df["intensity"]), dims="obs_id"
            )
            durations = pymc.MutableData(
                "durations", pymc.floatX(empty_df["duration"]), dims="obs_id"
            )
            mean_durations = pymc.MutableData(
                "mean_durations", pymc.floatX(empty_df["duration"]), dims="obs_id"
            )
            pains = pymc.MutableData(
                "pains", pymc.floatX(empty_df["duration"]), dims="obs_id"
            )
            mu += type_intercept[types]

            intensity_coefficients = pymc.Normal(
                "intensity_coefficients", mu=0, sigma=1, shape=2
            )
            duration_coefficients = pymc.Normal(
                "duration_coefficients", mu=0, sigma=1, shape=2
            )
            pain_coefficients = pymc.Normal("pain_coefficients", mu=0, sigma=1, shape=2)
            mu += (
                intensity_coefficients[0] + intensity_coefficients[1] * mean_intensities
            ) * intensities
            mu += (
                duration_coefficients[0] + duration_coefficients[1] * mean_durations
            ) * durations
            mu += (
                (pain_coefficients[0] + pain_coefficients[1] * pains)
                * intensities
                * durations
            )

            observed_outcomes = pymc.MutableData(
                "observed_outcomes", empty_df["pain_reduction"], dims="obs_id"
            )
            outcome = pymc.Normal(
                "outcome",
                mu=mu,
                sigma=pymc.Exponential(name="sigma", lam=1),
                observed=observed_outcomes,
                dims="obs_id",
            )

    def update_posterior(self, history_df, _):
        if not self.model:
            self.setup_model()

        with self.model:
            pymc.set_data(
                {
                    "types": history_df["type"],
                    "durations": history_df["duration"],
                    "mean_durations": history_df["mean_duration"],
                    "intensities": history_df["intensity"],
                    "mean_intensities": history_df["mean_intensity"],
                    "pains": history_df["current_pain"],
                    "observed_outcomes": history_df["pain_reduction"],
                }
            )
            self.trace = pymc.sample(2000, progressbar=False)

    def approximate_max_probabilities(self, number_of_treatments, context):
        assert (
            self.trace is not None
        ), "You called `approximate_max_probabilites` without updating the posterior"

        df = pandas.DataFrame(
            [context] * number_of_treatments,
        )
        df["activity_index"] = range(number_of_treatments)
        activity_df = pandas.DataFrame(self.possible_actions)

        df = pandas.concat([df, pandas.DataFrame(self.possible_actions)], axis=1)

        # Eliminate duplicate columns
        df = df.loc[:, ~df.columns.duplicated()].copy()

        with self.model:
            pymc.set_data(
                {
                    "types": df["type"],
                    "durations": df["duration"],
                    "mean_durations": df["mean_duration"],
                    "intensities": df["intensity"],
                    "mean_intensities": df["mean_intensity"],
                    "pains": df["current_pain"],
                }
            )
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
