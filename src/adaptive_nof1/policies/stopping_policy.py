from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import List
from typing_extensions import override

from adaptive_nof1.policies.policy import Policy
from adaptive_nof1.basic_types import Context, History


class StoppingPolicy(Policy):
    def __init__(self, policy, stopping_time):
        self.policy = policy
        self.stopping_time: Callable[[History, Context], bool] = stopping_time
        self._debug_information = []
        self._debug_data = []
        self._is_stopped = False

    @property
    def debug_information(self) -> List[str]:
        return self.policy.debug_information

    @override
    def choose_action(self, history, context, block_length=1):
        self._is_stopped = self.stopping_time(history, context)
        return self.policy.choose_action(history, context, block_length)

    def available_actions(self):
        return self.policy.available_actions()

    @property
    def debug_data(self) -> List[dict]:
        return self.policy.debug_data

    def get_policy_by_name(self, name):
        if str(self) == name:
            return self

    def __str__(self):
        return f"StoppingPolicy({self.policy})"

    @property
    def is_stopped(self):
        return self._is_stopped
