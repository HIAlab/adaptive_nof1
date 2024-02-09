from abc import ABC, abstractmethod
from typing import List


class Policy(ABC):
    def __init__(
        self,
        outcome_name="outcome",
        treatment_name="treatment",
        number_of_actions=None,
    ):
        self._number_of_actions = number_of_actions
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
        return []

    @property
    def number_of_actions(self):
        return self._number_of_actions

    @number_of_actions.setter
    def number_of_actions(self, value):
        self._number_of_actions = value

    @property
    def debug_data(self) -> List[dict]:
        return self._debug_data

    @property
    def is_stopped(self):
        return False

    @property
    def additional_config(self):
        return {}

    def get_policy_by_name(self, name):
        if str(self) == name:
            return self

        return False
