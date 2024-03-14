from adaptive_nof1.policies.policy import Policy
from adaptive_nof1.helpers import series_to_indexed_array, array_almost_equal

import numpy

import random


# This is a re-implementation of the stabilizing technique described in
# Shrestha S, Jain S. A Bayesian‐bandit adaptive design for N‐of‐1 clinical trials. Statistics in Medicine. 2021;40(7):1825-1844. doi:10.1002/sim.8873


class StabilizedThompsonSampling(Policy):
    def __init__(
        self,
        inference_model,
        length,
        posterior_update_interval=1,
        **kwargs,
    ):
        self.inference = inference_model
        self.posterior_update_interval = posterior_update_interval
        self._debug_data = []
        self.length = length
        super().__init__(**kwargs)

    def __str__(self):
        return f"StabilizedThompsonSampling({self.inference})"

    def probability_array(self, context):
        return self.inference.approximate_max_probabilities(
            self.number_of_actions, context
        )

    def stabilize_probabilities(self, probabilities, c):
        exponentiated_probabilies = [p**c for p in probabilities]
        summed_probabilitites = sum(exponentiated_probabilies)
        return [p / summed_probabilitites for p in exponentiated_probabilies]

    def choose_action(self, history, context, block_length=None):
        if (
            len(history) % self.posterior_update_interval == 0
            or self.inference.trace is None
        ):
            self.inference.update_posterior(history, self.number_of_actions)

        assert (
            block_length == None
        ), "Currently not implemented to work with stabilizing"

        probability_array = self.probability_array(context)

        # Standard value used in the paper, will range from 0 to 0.5
        c = context["t"] / 2 * self.length
        stabilized_probabilities = self.stabilize_probabilities(probability_array, c)

        action = random.choices(
            range(self.number_of_actions),
            weights=stabilized_probabilities,
        )[0]

        self._debug_information += [
            f"Probabilities for picking: {numpy.array_str(numpy.array(stabilized_probabilities), precision=2, suppress_small=True)}, chose {action}"
        ]
        self._debug_data.append(
            {
                "probabilities": probability_array,
                "stabilized_probabilities": stabilized_probabilities,
            }
        )
        return {self.treatment_name: action}

    @property
    def debug_data(self):
        return self._debug_data
