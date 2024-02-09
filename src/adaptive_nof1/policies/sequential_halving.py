from adaptive_nof1.policies.policy import Policy

import numpy
import math

from itertools import cycle, islice


class SequentialHalving(Policy):
    def __init__(
        self,
        inference_model,
        length,
        block_length=1,
        **kwargs,
    ):
        self.length = length
        self.block_length = block_length
        self.inference = inference_model
        self._debug_data = []
        self.treatments_to_consider = None
        self._number_of_interventions = None
        self.sequence_for_period = None
        self.rng = numpy.random.default_rng()
        super().__init__(**kwargs)

    @property
    def number_of_interventions(self):
        return self._number_of_interventions

    @number_of_interventions.setter
    def number_of_interventions(self, value):
        self._number_of_interventions = value
        if value is not None:
            self.treatments_to_consider = list(range(value))
            self.number_of_periods = math.ceil(math.log2(value))
            self.period_length = (
                self.length // self.block_length // self.number_of_periods
            )

    @property
    def number_of_actions(self):
        return self._number_of_interventions

    @number_of_actions.setter
    def number_of_actions(self, value):
        self.number_of_interventions = value

    def __str__(self):
        return f"SequentialHalving({self.length})"

    def is_first_in_period(self, context):
        return self.position_in_period(context) == 0

    def position_in_period(self, context):
        return (context["t"] // self.block_length) % self.period_length

    def repeat_list_to_period_length(self, lst):
        # Cycle through the list indefinitely and slice the first n elements
        return list(islice(cycle(lst), self.period_length))

    def k_lowest_indices(self, probabilities, k):
        probs_array = numpy.array(probabilities)
        sorted_indices = numpy.argsort(probs_array)
        return sorted_indices[:k]

    def choose_action(self, history, context):
        self.rng = numpy.random.default_rng()

        self.inference.update_posterior(history, self.number_of_actions)
        inference_debug_data = self.inference.debug_data
        self._debug_information += [""]

        if self.is_first_in_period(context):
            if context["t"] != 0:
                # Drop the lowest elements
                probabilities = self.inference.approximate_max_probabilities(
                    self.number_of_actions, context
                )

                k = len(self.treatments_to_consider) - math.ceil(
                    len(self.treatments_to_consider) / 2
                )
                to_remove = self.k_lowest_indices(probabilities, k)
                self.treatments_to_consider = [
                    item
                    for item in self.treatments_to_consider
                    if item not in to_remove
                ]

            # generate a random permutation of treatments
            self.sequence_for_period = self.repeat_list_to_period_length(
                list(self.rng.permutation(self.treatments_to_consider))
            )

        self._debug_data += [
            {**inference_debug_data, "sequence": self.sequence_for_period}
        ]
        return {
            self.treatment_name: self.sequence_for_period[
                self.position_in_period(context)
            ]
        }

    @property
    def debug_data(self):
        return self._debug_data
