from abc import ABC, abstractmethod


class AbstractPolicy(ABC):
    def __init__(self, number_of_actions):
        self.number_of_actions = number_of_actions

    @abstractmethod
    def choose_action(self, history):
        pass


class ConstantPolicy(AbstractPolicy):
    def __init__(self, number_of_actions, action):
        self.action = action
        super.__init__(number_of_actions)

    def choose_action(self, _):
        return self.action
