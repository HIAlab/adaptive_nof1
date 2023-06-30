from adaptive_nof1.helpers import values_to_index, index_to_actions
from adaptive_nof1.policies.policy import Policy
import copy

from typing import List

from functools import reduce
import operator


class CrossoverPolicy(Policy):
    def __init__(
        self,
        policy: Policy,
        action_dimensions: List[int],
        action_names: List[str],
    ):
        assert len(action_dimensions) == len(action_names)
        super().__init__(number_of_actions=action_dimensions)
        self.policy = policy
        self.action_dimensions = action_dimensions
        self.action_names = action_names
        self.max_index = reduce(operator.mul, action_dimensions)

    def __str__(self):
        return f"CrossoverPolicy({self.policy})"

    def actions_to_index(self, actions):
        return values_to_index(self.action_dimensions, list(actions.values()))

    def index_to_actions(self, index):
        return index_to_actions(index, self.action_dimensions, self.action_names)

    def transform_history(self, history):
        for observation in copy.deepcopy(history).observations:
            actions = observation.treatment
            actions["treatment"] = self.actions_to_index(actions)
        return history

    def choose_action(self, history, context):
        action = self.policy.choose_action(self.transform_history(history), context)[
            self.policy.treatment_name
        ]
        action_index = action - 1
        self._debug_information += [f"CrossoverIndex: {action_index}"]
        return self.index_to_actions(action_index)

    def available_actions(self):
        return [self.index_to_actions(index) for index in range(self.max_index)]
