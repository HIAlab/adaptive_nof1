from adaptive_nof1.inference.bayes import GaussianAverageTreatmentEffect
from adaptive_nof1.policies.policy import Policy

import numpy

import random


class UpperConfidenceBound(Policy):
    def __init__(self, epsilon: float, inference_model, **kwargs):
        self.epsilon = epsilon
        self.inference = inference_model
        super().__init__(**kwargs)

    def __str__(self):
        return f"UpperConfidenceBound({self.epsilon} epsilon, {self.inference})"

    def choose_best_action(self, history):
        outcome_groupby = history.to_df().groupby(self.treatment_name)["outcome"].mean()
        best_row = outcome_groupby.idxmin()
        return best_row

    @property
    def additional_config(self):
        return {"inference": f"{self.inference}"}

    def choose_action(self, history, context):
        self.inference.update_posterior(history, self.number_of_actions)
        self.inference.approximate_max_probabilities(self.number_of_actions, context)
        upper_bounds_array = self.inference.get_upper_confidence_bounds(
            self.outcome_name
        )
        self.debug_data.append(
            {"upper_bounds_array": upper_bounds_array, **self.inference.debug_data}
        )

        if context["t"] == 0:
            self._debug_information += ["First Action"]
            return {
                self.treatment_name: random.choices(range(self.number_of_actions))[0]
            }
        # Todo: make debug data work without this call
        self._debug_information += [""]
        return {self.treatment_name: numpy.argmax(upper_bounds_array)}


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

    @property
    def additional_config(self):
        return {"inference": f"{self.inference}"}

    def __str__(self):
        return f"ThompsonSampling({self.inference})"

    def choose_action(self, history, context):
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
            f"Probabilities for picking: {numpy.array_str(numpy.array(probability_array), precision=2, suppress_small=True)}, chose {action}"
        ]
        debug_data_from_model = self.inference.debug_data
        self._debug_data.append(
            {**{"probabilities": probability_array}, **debug_data_from_model}
        )
        return {self.treatment_name: action}

    @property
    def debug_data(self):
        return self._debug_data


class ClippedThompsonSampling(ThompsonSampling):
    def __str__(self):
        return f"ClippedThompsonSampling({self.inference})"

    def choose_action(self, history, context):
        if len(history) == 0:
            self._debug_information += ["len(History) == 0"]
            self._debug_data.append({})
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
            0.2,
            0.8,
        )
        action = random.choices(
            range(self.number_of_actions), weights=probability_array
        )[0]
        self._debug_information += [
            f"Probabilities for picking: {numpy.array_str(probability_array, precision=2, suppress_small=True)}, chose {action}"
        ]
        self._debug_data.append({"probabilities": probability_array})
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
            action_index = action
            probability_array[action_index] -= 0.2
        action = random.choices(
            range(self.number_of_actions), weights=probability_array
        )[0]
        self._debug_information += [
            f"Probabilities for picking: {numpy.array_str(probability_array, precision=2, suppress_small=True)}, chose {action}"
        ]
        return {self.treatment_name: action}
