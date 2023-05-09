from abc import ABC, abstractmethod
import random
import numpy as np

from src.bayes import *


class Policy(ABC):
    def __init__(self, number_of_actions):
        self.number_of_actions = number_of_actions
        self.debug_information = ""

    @abstractmethod
    def choose_action(self, history, context):
        pass


class ConstantPolicy(Policy):
    def __init__(self, number_of_actions, action):
        self.action = action
        super().__init__(number_of_actions)

    def __str__(self):
        return f"ConstantPolicy: {self.action}.\n"

    def choose_action(self, _, __):
        return self.action


class FixedPolicy(Policy):
    def __init__(self, number_of_actions, block_length):
        self.block_length = block_length
        super().__init__(number_of_actions)

    def __str__(self):
        return f"FixedPolicy: {self.block_length} block length.\n"

    def choose_action(self, history, _):
        round = len(history) // self.block_length
        return round % self.number_of_actions + 1


class FrequentistExploreThenCommit(Policy):
    def __init__(self, number_of_actions, block_length, explore_blocks=5):
        self.fixed_policy = FixedPolicy(number_of_actions, block_length)
        self.explore_blocks = explore_blocks

        self.best_treatment = None

        super().__init__(number_of_actions)

    def __str__(self):
        return f"FrequentistExploreThenCommit:{self.explore_blocks} explore blocks.\n"

    def choose_best_action(self, history):
        outcome_groupby = history.to_df().groupby("treatment")["outcome"].mean()
        best_row = outcome_groupby.idxmax()
        return best_row

    def choose_action(self, history, _):
        if self.best_treatment:
            return self.best_treatment
        if len(history) >= self.explore_blocks * self.fixed_policy.block_length:
            self.best_treatment = self.choose_best_action(history)
            return self.best_treatment
        return self.fixed_policy.choose_action(history, _)


class FrequentistEpsilonGreedy(Policy):
    def __init__(self, number_of_actions: int, block_length: int, epsilon: float):
        self.epsilon = epsilon
        self.block_length = block_length
        self.last_action = None
        super().__init__(number_of_actions)

    def __str__(self):
        return f"FrequentistEpsilonGreedy: {self.epsilon} epsilon"

    def choose_best_action(self, history):
        outcome_groupby = history.to_df().groupby("treatment")["outcome"].mean()
        best_row = outcome_groupby.idxmax()
        return best_row

    def choose_action(self, history, _):
        if len(history) < self.number_of_actions * self.block_length:
            return len(history) // self.block_length + 1

        if len(history) % self.block_length == 0:
            if random.random() < self.epsilon:
                self.last_action = random.choice(range(self.number_of_actions)) + 1
                return self.last_action

            self.last_action = self.choose_best_action(history)
            return self.last_action

        return self.last_action


class UpperConfidenceBound(Policy):
    def __init__(self, number_of_actions: int, block_length: int, epsilon: float):
        self.epsilon = epsilon
        self.inference = BayesianInference()
        self.block_length = block_length
        self.last_action = None
        super().__init__(number_of_actions)

    def __str__(self):
        return f"UpperConfidenceBound: {self.epsilon} epsilon"

    def choose_best_action(self, history):
        outcome_groupby = history.to_df().groupby("treatment")["outcome"].mean()
        best_row = outcome_groupby.idxmin()
        return best_row

    def choose_action(self, history, _):
        if len(history) % self.block_length == 0:
            self.inference.update_posterior(history, self.number_of_actions)
            bounds = (
                self.inference.get_upper_confidence_bounds("average_treatment_effect")
                .get("average_treatment_effect")
                .data
            )
            upper_bounds_array = [bound[1] for bound in bounds]
            self.last_action = np.argmax(upper_bounds_array) + 1
            self.debug_information += (
                f"Round {len(history)}: Upper Bounds Array: {upper_bounds_array}\n"
            )
            return self.last_action
        return self.last_action
