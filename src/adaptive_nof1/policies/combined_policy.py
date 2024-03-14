from adaptive_nof1.helpers import split_with_postfix
from adaptive_nof1.policies.policy import Policy

import numpy
from typing import List

import matplotlib.pyplot as pyplot
import itertools
from adaptive_nof1.helpers import flatten


class CombinedPolicy(Policy):
    def __init__(
        self,
        policies: List[Policy],
        split_context=False,
        update_context_with_actions=True,
        treatment_names: List[str] | None = None,
    ):
        super().__init__()
        self.policies = policies
        if treatment_names:
            for name, policy in zip(treatment_names, policies):
                policy.treatment_name = name
        self.split_context = split_context
        self.update_context_with_actions = update_context_with_actions

    def __str__(self):
        return f"CombinedPolicy({[str(policy) for policy in self.policies]})"

    @property
    def number_of_actions(self):
        return [policy.number_of_actions for policy in self.policies]

    @number_of_actions.setter
    def number_of_actions(self, values):
        for policy, value in zip(self.policies, values):
            policy.number_of_actions = value

    @property
    def debug_information(self):
        return list(zip(*[policy.debug_information for policy in self.policies]))

    @property
    def debug_data(self):
        return flatten([policy.debug_data for policy in self.policies])

    def choose_action(self, history, context):
        if self.split_context:
            contexts = split_with_postfix(context)
        else:
            contexts = [context] * len(self.policies)
        actions = {}
        for policy, context in zip(self.policies, contexts):
            if self.update_context_with_actions:
                context.update(actions)
            actions.update(policy.choose_action(history, context))
        return actions

    def available_actions(self):
        # Todo Implementation:
        return []
