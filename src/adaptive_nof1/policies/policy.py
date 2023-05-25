from abc import ABC, abstractmethod
from typing import List


class Policy(ABC):
    def __init__(self, number_of_actions):
        self.number_of_actions = number_of_actions
        self._debug_information = []

    @property
    def debug_information(self) -> List[str]:
        return self._debug_information

    @abstractmethod
    def choose_action(self, history, context, block_length=1):
        pass


