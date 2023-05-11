from abc import ABC, abstractmethod


class Policy(ABC):
    def __init__(self, number_of_actions):
        self.number_of_actions = number_of_actions
        self.debug_information = []

    @abstractmethod
    def choose_action(self, history, context, block_length=1):
        pass


