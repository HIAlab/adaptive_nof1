import numpy
from adaptive_nof1.models.model import Model

import scipy.stats as stats

from typing import List


class SelfExperimentationModel(Model):
    def __init__(
        self,
        patient_id,
        intervention_effects,
        baseline_drift=False,
        baseline_correlation=0.9,
        baseline_variance=1,
        correlation=0,
        flare_probability=0,
        decision_boundaries: None | List[float] = None,
        **kwargs,
    ):
        super().__init__(patient_id, **kwargs)
        self.rng = numpy.random.default_rng(self.patient_id)
        self.baseline_drift = baseline_drift
        self.baseline_correlation = baseline_correlation
        self.baseline_variance = baseline_variance

        self.correlation = correlation

        self.flare_probability = flare_probability

        self.decision_boundaries = decision_boundaries
        if decision_boundaries is None:
            # Equally split a normal (0, 1) distribution into 5 parts:
            self.decision_boundaries = numpy.array(
                [stats.norm.ppf(p) for p in [0.2, 0.4, 0.6, 0.8]]
            )

        self.intervention_effects = intervention_effects

    def generate_context(self, history):
        if len(history) == 0:
            return {}
        observation = history.observations[-1]
        return {
            "previous_context": observation.context,
            "previous_outcome": observation.outcome,
        }

    def observe_outcome(self, action, context):
        outcome = {}
        previous_outcome = {"continuous_outcome_before_flare": 0, "baseline": 0}
        if "previous_outcome" in context:
            previous_outcome = context["previous_outcome"]

        continuous_outcome = 0

        # Baseline
        if self.baseline_drift:
            # Assumed AR(1) with mean = 0
            baseline = self.baseline_correlation * previous_outcome[
                "baseline"
            ] + self.rng.normal(0, self.baseline_variance)
            outcome["baseline"] = baseline
            continuous_outcome += baseline

        # Correlation
        continuous_outcome += (
            self.correlation + previous_outcome["continuous_outcome_before_flare"]
        )
        outcome["continuous_outcome_before_flare"] = continuous_outcome

        # Intervention Effect
        continuous_outcome += self.intervention_effects[action["treatment"]]

        # Discretization to Ordinal values
        outcome["discretized_outcome"] = numpy.searchsorted(
            self.decision_boundaries,
            continuous_outcome,
        )

        # Flare
        if self.bernoulli(self.flare_probability):
            outcome["discretized_outcome"] = 1

        outcome["outcome"] = outcome["discretized_outcome"]

        return outcome

    def bernoulli(self, p):
        return self.rng.choice([True, False], p=[p, 1 - p])

    def __str__(self):
        return f"SelfExperimentationModel"
