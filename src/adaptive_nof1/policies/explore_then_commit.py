from adaptive_nof1.policies.policy import Policy
from adaptive_nof1.helpers import series_to_indexed_array, array_almost_equal

import numpy


class ExploreThenCommit(Policy):
    def __init__(
        self,
        inference_model,
        exploration_length,
        block_length=1,
        randomize=False,
        **kwargs,
    ):
        self.exploration_length = exploration_length
        self.block_length = block_length
        self.inference = inference_model
        self.chosen_treatment = None
        self._debug_data = []
        self.treatment_sequence = None
        self.randomize = randomize
        super().__init__(**kwargs)

    def __str__(self):
        return f"ExploreThenCommit({self.exploration_length},{self.inference})"

    @Policy.number_of_actions.setter
    def number_of_actions(self, value):
        self._number_of_actions = value
        self.treatment_sequence = list(range(value))
        if self.randomize:
            numpy.random.default_rng().shuffle(self.treatment_sequence)

    @property
    def additional_config(self):
        return {"inference": f"{self.inference}"}

    def choose_action(self, history, context):
        self.inference.update_posterior(history, self.number_of_actions)
        inference_debug_data = self.inference.debug_data

        if self.chosen_treatment is not None:
            self._debug_information += ["Commit"]
            self._debug_data += [{"is_explore": False, **inference_debug_data}]
            return {self.treatment_name: self.chosen_treatment}

        if context["t"] / self.block_length >= self.exploration_length:
            probability_array = self.inference.approximate_max_probabilities(
                self.number_of_actions, context
            )
            self.chosen_treatment = numpy.argmax(probability_array)
            self._debug_data += [{"is_explore": False, **inference_debug_data}]
            return {self.treatment_name: self.chosen_treatment}

        self._debug_information += ["Explore"]
        self._debug_data += [{"is_explore": True, **inference_debug_data}]
        return {
            self.treatment_name: self.treatment_sequence[
                (context["t"] // self.block_length) % self.number_of_actions
            ]
        }

    @property
    def debug_data(self):
        return self._debug_data
