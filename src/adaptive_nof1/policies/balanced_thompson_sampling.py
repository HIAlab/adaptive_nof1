from adaptive_nof1.policies.policy import Policy
from adaptive_nof1.helpers import series_to_indexed_array

import numpy

import random


class BalancedThompsonSampling(Policy):
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
        return f"BalancedThompsonSampling({self.inference})"

    def choose_action(self, history, context, block_length=None):
        if (
            len(history) % self.posterior_update_interval == 0
            or self.inference.trace is None
        ):
            self.inference.update_posterior(history, self.number_of_actions)

        probability_array = self.inference.approximate_max_probabilities(
            self.number_of_actions, context
        )

        treatment_counts_array = series_to_indexed_array(
            history.to_df()
            .groupby(self.treatment_name, group_keys=True)[self.treatment_name]
            .count(),
            min_length=self.number_of_actions,
        )
        sum_of_treatment_counts = sum(treatment_counts_array)

        if sum_of_treatment_counts != 0:
            relative_frequencies_array = [
                count / sum_of_treatment_counts for count in treatment_counts_array
            ]
        else:
            relative_frequencies_array = [0] * self.number_of_actions

        if all(relative_frequencies_array == probability_array):
            action = random.choices(
                range(self.number_of_actions), weights=probability_array
            )[0]
        else:
            subtraction = [
                relative - probability
                for relative, probability in zip(
                    relative_frequencies_array, probability_array
                )
            ]
            action = numpy.argmin(subtraction)

        self._debug_information += [
            f"Probabilities for picking: {numpy.array_str(numpy.array(probability_array), precision=2, suppress_small=True)}, relative_frequencies: {numpy.array_str(numpy.array(relative_frequencies_array), precision=2, suppress_small=True)}, chose {action}"
        ]
        self._debug_data.append(
            {
                "probabilities": probability_array,
                "relative_frequencies": relative_frequencies_array,
            }
        )
        return {self.treatment_name: action}

    @property
    def debug_data(self):
        return self._debug_data
