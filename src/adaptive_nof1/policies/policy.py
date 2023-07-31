from abc import ABC, abstractmethod
from typing import List


class Policy(ABC):
    def __init__(
        self, number_of_actions, outcome_name="outcome", treatment_name="treatment"
    ):
        self.number_of_actions = number_of_actions
        self.outcome_name = outcome_name
        self.treatment_name = treatment_name
        self._debug_information = []
        self._debug_data = []

    @property
    def debug_information(self) -> List[str]:
        return self._debug_information

    @abstractmethod
    def choose_action(self, history, context, block_length=1):
        pass

    def available_actions(self):
        return [
            {self.treatment_name: action}
            for action in range(1, self.number_of_actions + 1)
        ]

    @property
    def debug_data(self) -> List[dict]:
        return self._debug_data

    def get_policy_by_name(self, name):
        if str(self) == name:
            return self
