from adaptive_nof1.policies.policy import Policy

import numpy
from typing import List

import matplotlib.pyplot as pyplot
import itertools


class ComposedPolicy(Policy):
    def __init__(
        self,
        policies: List[Policy],
        durations: List[int],
        block_length=None,
        treatment_name: str | None = None,
        **kwargs,
    ):
        self.policies = policies

        self.switch_points = numpy.cumsum(durations)
        self.current_index = 0
        self.block_length = block_length
        self._treatment_name = treatment_name
        super().__init__(**kwargs)

    @property
    def treatment_name(self):
        return self._treatment_name

    @treatment_name.setter
    def treatment_name(self, name):
        self._treatment_name = name
        for policy in self.policies:
            policy.treatment_name = name

    def __str__(self):
        return f"ComposedPolicy({[str(policy) for policy in self.policies]})"

    @property
    def debug_information(self):
        return [info for policy in self.policies for info in policy.debug_information]

    def choose_action(self, history, context):
        if len(history) > self.switch_points[
            self.current_index
        ] and self.current_index + 1 < len(self.policies):
            self.current_index += 1

        current_policy = self.policies[self.current_index]
        return current_policy.choose_action(
            history,
            context,
        )

    def plot(self):
        colors = ["black", "grey"]
        pyplot.axvspan(0, self.switch_points[0], facecolor=colors[1], alpha=0.1)
        for index, [first, second] in enumerate(itertools.pairwise(self.switch_points)):
            pyplot.axvspan(
                first, second, facecolor=colors[index % len(colors)], alpha=0.1
            )

    def available_actions(self):
        return []
