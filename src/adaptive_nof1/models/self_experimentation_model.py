from joblib.externals.cloudpickle.cloudpickle_fast import _property_reduce
import numpy
from adaptive_nof1.models.model import Model
from adaptive_nof1.helpers import contains_keys

import scipy.stats as stats

from typing import List


class SelfExperimentationModel(Model):
    def __init__(
        self,
        patient_id,
        intervention_effects,
        baseline_model="",
        baseline_config={},
        correlation=0,
        spike_probability=0,
        decision_boundaries: None | List[float] = None,
        **kwargs,
    ):
        super().__init__(patient_id, **kwargs)
        self.rng = numpy.random.default_rng(self.patient_id)
        self.baseline_model = baseline_model
        self.baseline_config = baseline_config

        self.correlation = correlation

        self.spike_probability = spike_probability

        self.decision_boundaries = decision_boundaries
        if decision_boundaries is None:
            # Equally split a normal (0, 1) distribution into 5 parts:
            self.decision_boundaries = numpy.array(
                [stats.norm.ppf(p) for p in [0.2, 0.4, 0.6, 0.8]]
            )

        self.intervention_effects = intervention_effects

        if self.baseline_model == "linear":
            assert contains_keys(self.baseline_config, ["start", "end", "min", "max"])
        elif self.baseline_model == "random_walk":
            assert contains_keys(
                self.baseline_config, ["min", "max", "variance", "correlation"]
            )
        elif self.baseline_model == "noise":
            assert contains_keys(self.baseline_config, ["variance"])
        elif self.baseline_model == "":
            self.baseline_config["name"] = ""
        else:
            raise AssertionError(f"Unknown baseline model: {self.baseline_model}")

        if "name" not in self.baseline_config:
            self.baseline_config["name"] = self.baseline_model

    @property
    def additional_config(self):
        return {"true_intervention_effect": self.intervention_effects[1]}

    def generate_context(self, history):
        if len(history) == 0:
            return {}
        observation = history.observations[-1]
        return {
            "previous_context": observation.context,
            "previous_outcome": observation.outcome,
        }

    def calculate_baseline(self, context, previous_outcome):
        # Baseline
        if self.baseline_model == "random_walk":
            # Assumed AR(1) with mean = 0
            baseline = self.baseline_config["correlation"] * previous_outcome[
                "baseline"
            ] + self.rng.normal(0, self.baseline_config["variance"])

            # Bound baseline to [-1;1]
            baseline = max(
                self.baseline_config["min"], min(self.baseline_config["max"], baseline)
            )
            return baseline

        if self.baseline_model == "linear":
            t = context["t"]
            if t < self.baseline_config["start"]:
                return self.baseline_config["min"]

            if t > self.baseline_config["end"]:
                return self.baseline_config["max"]

            slope = (self.baseline_config["max"] - self.baseline_config["min"]) / (
                self.baseline_config["end"] - self.baseline_config["start"]
            )
            return (t - self.baseline_config["start"]) * slope + self.baseline_config[
                "min"
            ]

        if self.baseline_model == "noise":
            return self.rng.normal(0, self.baseline_config["variance"])

        return 0

    def observe_outcome(self, action, context):
        outcome = {}
        previous_outcome = {"continuous_outcome": 0, "baseline": 0}
        if "previous_outcome" in context:
            previous_outcome = context["previous_outcome"]

        baseline = self.calculate_baseline(context, previous_outcome)

        outcome["baseline"] = baseline
        continuous_outcome = baseline

        # Intervention Effect
        continuous_outcome += self.intervention_effects[action["treatment"]]

        # Correlation
        continuous_outcome = (
            self.correlation * previous_outcome["continuous_outcome"]
            + (1 - self.correlation) * continuous_outcome
        )

        outcome["continuous_outcome"] = continuous_outcome

        # Discretization to Ordinal values
        outcome["discretized_outcome"] = numpy.searchsorted(
            self.decision_boundaries,
            continuous_outcome,
        )

        # Flare
        if self.bernoulli(self.spike_probability):
            outcome["discretized_outcome"] = 1

        outcome["outcome"] = outcome["discretized_outcome"]

        return outcome

    def bernoulli(self, p):
        return self.rng.choice([True, False], p=[p, 1 - p])

    def __str__(self):
        return f"SEM[{self.intervention_effects}, {self.baseline_config['name']}, {self.correlation}, {self.spike_probability}]"
