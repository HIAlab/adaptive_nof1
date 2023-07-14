from adaptive_nof1.policies.policy import Policy
from typing import List

import copy


class SelectionPolicy(Policy):
    def __init__(
        self,
        possible_actions: List[dict],
        columns: List[str],
        policy: Policy,
    ):
        self.columns = columns
        self.possible_actions = possible_actions
        self.policy = policy

        super().__init__(number_of_actions=len(possible_actions))

    def __str__(self):
        return f"SelectionPolicy{self.policy}"

    def selection_index_treatment_name(self):
        return "selection_index"

    def index_to_actions(self, selection_index):
        return self.possible_actions[selection_index]

    def transform_history(self, history):
        history = copy.deepcopy(history)
        for observation in history.observations:
            actions = observation.treatment
            actions.update(self.possible_actions[actions["selection_index"]])
        return history

    @property
    def debug_information(self):
        return list(zip(self._debug_information, self.policy.debug_information))

    def choose_action(self, history, context):
        action = self.policy.choose_action(self.transform_history(history), context)[
            self.policy.treatment_name
        ]
        selection_index = action - 1
        self._debug_information += [
            f"{self.selection_index_treatment_name()}: {selection_index}"
        ]
        return {
            **self.index_to_actions(selection_index),
            self.selection_index_treatment_name(): selection_index,
        }

    def available_actions(self):
        return [
            {
                **self.index_to_actions(selection_index),
                self.selection_index_treatment_name(): selection_index,
            }
            for selection_index in range(self.number_of_actions)
        ]

    def get_policy_by_name(self, name):
        if str(self) == name:
            return self
        return self.policy.get_policy_by_name(name)
