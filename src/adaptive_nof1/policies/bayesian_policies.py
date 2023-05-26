from adaptive_nof1.inference.bayes import GaussianAverageTreatmentEffect
from adaptive_nof1.policies.policy import Policy

import numpy

import random


class UpperConfidenceBound(Policy):
    def __init__(self, number_of_actions: int, epsilon: float):
        self.epsilon = epsilon
        self.inference = GaussianAverageTreatmentEffect()
        super().__init__(number_of_actions)

    def __str__(self):
        return f"UpperConfidenceBound: {self.epsilon} epsilon"

    def choose_best_action(self, history):
        outcome_groupby = history.to_df().groupby("treatment")["outcome"].mean()
        best_row = outcome_groupby.idxmin()
        return best_row

    def choose_action(self, history, _, block_length):
        self.inference.update_posterior(history, self.number_of_actions)
        bounds = (
            self.inference.get_upper_confidence_bounds("average_treatment_effect")
            .get("average_treatment_effect")
            .data
        )
        upper_bounds_array = [bound[1] for bound in bounds]
        self.debug_information += [
            f"Round {len(history)}: Upper Bounds Array: {upper_bounds_array}"
        ]
        return np.argmax(upper_bounds_array) + 1


class ThompsonSampling(Policy):
    def __init__(
        self, number_of_actions: int, inference_model, posterior_update_interval=1
    ):
        super().__init__(number_of_actions)
        self.inference = inference_model
        self.posterior_update_interval = posterior_update_interval

    def __str__(self):
        return f"ThompsonSampling({self.inference})"

    def choose_action(self, history, _, block_length=None):
        if len(history) % self.posterior_update_interval == 0 or not hasattr(
            self.inference, "trace"
        ):
            self.inference.update_posterior(history, self.number_of_actions)
        probability_array = self.inference.approximate_max_probabilities(
            self.number_of_actions
        )
        action = (
            random.choices(range(self.number_of_actions), weights=probability_array)[0]
            + 1
        )
        self._debug_information += [
            f"Probabilities for picking: {numpy.array_str(probability_array, precision=2, suppress_small=True)}, chose {action}"
        ]
        return action


class ClippedThompsonSampling(Policy):
    def __init__(
        self, number_of_actions: int, inference_model, posterior_update_interval=1
    ):
        super().__init__(number_of_actions)
        self.inference = inference_model
        self.posterior_update_interval = posterior_update_interval

    def __str__(self):
        return f"ClippedThompsonSampling({self.inference})"

    def choose_action(self, history, _, block_length=None):
        if len(history) % self.posterior_update_interval == 0 or not hasattr(
            self.inference, "trace"
        ):
            self.inference.update_posterior(history, self.number_of_actions)
        probability_array = numpy.clip(self.inference.approximate_max_probabilities(
            self.number_of_actions
        ), 0.1, 0.9)
        action = (
            random.choices(range(self.number_of_actions), weights=probability_array)[0]
            + 1
        )
        self._debug_information += [
            f"Probabilities for picking: {numpy.array_str(probability_array, precision=2, suppress_small=True)}, chose {action}"
        ]
        return action
