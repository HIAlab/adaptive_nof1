import numpy as np
import random
from abc import ABC, abstractmethod

from src.adaptive_nof1.inference.bayes import *
from src.adaptive_nof1.policies.fixed_policy import FixedPolicy


class Policy(ABC):
    def __init__(self, number_of_actions):
        self.number_of_actions = number_of_actions
        self.debug_information = []

    @abstractmethod
    def choose_action(self, history, context, block_length=1):
        pass


