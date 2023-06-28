from adaptive_nof1.policies.policy import Policy

from typing import List

def index_to_values(dimensions, index):
    values = []
    for dimension in dimensions:
        values.append(index % dimension)
        index //= dimension
    values.reverse()
    
    return values

def values_to_index(dimensions, values):
    index = 0
    for i in range(len(dimensions)):
        index += values[i]
        if i < len(dimensions) - 1:
            index *= dimensions[i + 1]
    return index

class CrossoverPolicy(Policy):
    def __init__(
        self,
        policy: Policy,
        number_of_actions: List[int],
        treatment_names: List[str],
    ):
        assert len(number_of_actions) == len(treatment_names)
        super().__init__(number_of_actions=number_of_actions)
        self.policy = policy
        self.number_of_actions = number_of_actions
        self.treatment_names = treatment_names

    def __str__(self):
        return f"CrossoverPolicy({self.policy})"

    def actions_to_index(self, actions):
        return values_to_index(self.number_of_actions, actions.values())

    def index_to_actions(self, index):
        values = index_to_values(self.number_of_actions, index)
        return {treatment_name: action for treatment_name, action in zip(self.treatment_names, values)}


    def transform_history(self, history):
        for observation in history.observations:
            actions = observation.treatment
            actions["treatment"] = self.actions_to_index(actions)
        return history

    def choose_action(self, history, context):
        action_index = self.policy.choose_action(self.transform_history(history), context)["treatment"]
        self._debug_information += [f"CrossoverIndex: {action_index}"]
        return self.index_to_actions(action_index)

    def available_actions(self):
        # Todo Implementation:
        return []
