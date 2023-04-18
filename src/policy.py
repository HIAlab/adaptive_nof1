from abc import ABC, abstractmethod


class Policy(ABC):
    def __init__(self, number_of_actions):
        self.number_of_actions = number_of_actions

    @abstractmethod
    def choose_action(self, history, context):
        pass


class ConstantPolicy(Policy):
    def __init__(self, number_of_actions, action):
        self.action = action
        super().__init__(number_of_actions)

    def choose_action(self, _, __):
        return self.action


class FixedPolicy(Policy):
    def __init__(self, number_of_actions, block_length):
        self.block_length = block_length
        super().__init__(number_of_actions)

    def choose_action(self, history, _):
        round = len(history) // self.block_length
        return round % self.number_of_actions + 1
