from adaptive_nof1.inference.bayes import GaussianAverageTreatmentEffect
from adaptive_nof1.policies.policy import Policy

import numpy

import random


class UpperConfidenceBound(Policy):
    def __init__(self, number_of_actions: int, epsilon: float):
        self.epsilon = epsilon
        self.inference = GaussianAverageTreatmentEffect(
            treatment_name=self.treatment_name
        )
        super().__init__(number_of_actions)

    def __str__(self):
        return f"UpperConfidenceBound: {self.epsilon} epsilon"

    def choose_best_action(self, history):
        outcome_groupby = history.to_df().groupby(self.treatment_name)["outcome"].mean()
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
        self.debug_data.append({"upper_bounds_array": upper_bounds_array})
        return np.argmax(upper_bounds_array)


class ThompsonSampling(Policy):
    def __init__(
        self,
        inference_model,
        posterior_update_interval=1,
        **kwargs,
    ):
        self.inference = inference_model
        self.posterior_update_interval = posterior_update_interval
        self._debug_data = []
        super().__init__(**kwargs)

    def __str__(self):
        return f"ThompsonSampling({self.inference})"

    def choose_action(self, history, context, block_length=None):
        if (
            len(history) % self.posterior_update_interval == 0
            or self.inference.trace is None
        ):
            self.inference.update_posterior(history, self.number_of_actions)

        probability_array = self.inference.approximate_max_probabilities(
            self.number_of_actions, context
        )
        action = random.choices(
            range(self.number_of_actions), weights=probability_array
        )[0]
        self._debug_information += [
            f"Probabilities for picking: {numpy.array_str(probability_array, precision=2, suppress_small=True)}, chose {action}"
        ]
        self._debug_data.append({"probabilities": probability_array})
        return {self.treatment_name: action}

    @property
    def debug_data(self):
        return self._debug_data


class ClippedThompsonSampling(ThompsonSampling):
    def __str__(self):
        return f"ClippedThompsonSampling({self.inference})"

    def choose_action(self, history, context, block_length=None):
        if len(history) == 0:
            self._debug_information += ["len(History) == 0"]
            return {
                self.treatment_name: random.choices(range(self.number_of_actions))[0]
            }

        if (
            len(history) % self.posterior_update_interval == 0
            or self.inference.trace is None
        ):
            self.inference.update_posterior(history, self.number_of_actions)
        probability_array = numpy.clip(
            self.inference.approximate_max_probabilities(
                self.number_of_actions, context
            ),
            0.1,
            0.9,
        )
        action = random.choices(
            range(self.number_of_actions), weights=probability_array
        )[0]
        self._debug_information += [
            f"Probabilities for picking: {numpy.array_str(probability_array, precision=2, suppress_small=True)}, chose {action}"
        ]
        return {self.treatment_name: action}


class ClippedHistoryAwareThompsonSampling(ThompsonSampling):
    def __str__(self):
        return f"ClippedHistoryAwareThompsonSampling({self.inference})"

    def choose_action(self, history, context, block_length=None):
        if len(history) == 0:
            self._debug_information += ["len(History) == 0"]
            return {
                self.treatment_name: random.choices(range(self.number_of_actions))[0]
            }

        if (
            len(history) % self.posterior_update_interval == 0
            or self.inference.trace is None
        ):
            self.inference.update_posterior(history, self.number_of_actions)
        probability_array = numpy.clip(
            self.inference.approximate_max_probabilities(
                self.number_of_actions, context
            ),
            0.1,
            0.9,
        )
        last_three_actions = [
            observation.treatment[self.treatment_name]
            for observation in history.observations[-3:]
        ]

        # Penalize using the same action
        for action in last_three_actions:
            action_index = action - 1
            probability_array[action_index] -= 0.2
        action = random.choices(
            range(self.number_of_actions), weights=probability_array
        )[0]
        self._debug_information += [
            f"Probabilities for picking: {numpy.array_str(probability_array, precision=2, suppress_small=True)}, chose {action}"
        ]
        return {self.treatment_name: action}
