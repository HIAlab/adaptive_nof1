from adaptive_nof1.dict_helpers import split_with_postfix
from adaptive_nof1.policies.policy import Policy

import numpy
from typing import List

import matplotlib.pyplot as pyplot
import itertools


class CombinedPolicy(Policy):
    def __init__(
        self,
        policies: List[Policy],
        number_of_actions=None,
    ):
        super().__init__(number_of_actions=number_of_actions)
        self.policies = policies

    def __str__(self):
        return f"CombinedPolicy({[str(policy) for policy in self.policies]})"

    @property
    def debug_information(self):
        return [info for policy in self.policies for info in policy.debug_information]

    def choose_action(self, history, context):
        contexts = split_with_postfix(context)
        actions = {
            policy.treatment_name: policy.choose_action(history, context)
            for policy, context in zip(self.policies, contexts)
        }
        return actions

    def available_actions(self):
        # Todo Implementation:
        return []
