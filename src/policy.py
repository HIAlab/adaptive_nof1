from abc import ABC, abstractmethod
import random
import numpy as np

from src.bayes import *


class Policy(ABC):
    def __init__(self, number_of_actions):
        self.number_of_actions = number_of_actions
        self.debug_information = []

    @abstractmethod
    def choose_action(self, history, context, block_length=1):
        pass


class BlockPolicy(Policy):
    def __init__(self, internal_policy: Policy, block_length):
        self.internal_policy = internal_policy
        self.block_length = block_length
        self.last_action = None

    def __str__(self):
        return f"BlockPolicy({self.internal_policy})"

    def is_first_of_block(self, number):
        return number % self.block_length == 0

    def choose_action(self, history, context):
        if self.is_first_of_block(len(history)):
            self.last_action = self.internal_policy.choose_action(
                history, context, block_length=self.block_length
            )
        return self.last_action


class ConstantPolicy(Policy):
    def __init__(self, number_of_actions, action):
        self.action = action
        super().__init__(number_of_actions)

    def __str__(self):
        return f"ConstantPolicy: {self.action}.\n"

    def choose_action(self):
        return self.action


class FixedPolicy(Policy):
    def __init__(self, number_of_actions):
        super().__init__(number_of_actions)

    def __str__(self):
        return f"FixedPolicy\n"

    def choose_action(self, history, _, block_length=1):
        round = len(history) // block_length
        return round % self.number_of_actions + 1


class FrequentistExploreThenCommit(Policy):
    def __init__(self, number_of_actions, explore_blocks=5):
        self.fixed_policy = FixedPolicy(number_of_actions)
        self.explore_blocks = explore_blocks

        self.best_treatment = None

        super().__init__(number_of_actions)

    def __str__(self):
        return f"FrequentistExploreThenCommit:{self.explore_blocks} explore blocks.\n"

    def choose_best_action(self, history):
        outcome_groupby = history.to_df().groupby("treatment")["outcome"].mean()
        best_row = outcome_groupby.idxmax()
        return best_row

    def choose_action(self, history, _, block_length=1):
        if self.best_treatment:
            self.debug_information += ["Commit"]
            return self.best_treatment
        if len(history) >= self.explore_blocks * block_length:
            self.debug_information += ["Explore"]
            self.best_treatment = self.choose_best_action(history)
            return self.best_treatment
        return self.fixed_policy.choose_action(history, _, block_length)


class FrequentistEpsilonGreedy(Policy):
    def __init__(self, number_of_actions: int, epsilon: float):
        self.epsilon = epsilon
        self.last_action = None
        super().__init__(number_of_actions)

    def __str__(self):
        return f"FrequentistEpsilonGreedy: {self.epsilon} epsilon"

    def choose_best_action(self, history):
        outcome_groupby = history.to_df().groupby("treatment")["outcome"].mean()
        best_row = outcome_groupby.idxmax()
        return best_row

    def choose_action(self, history, _, block_length=1):
        if len(history) < self.number_of_actions * block_length:
            self.debug_information += ["Initial Round"]
            return len(history) // block_length + 1

        if random.random() < self.epsilon:
            self.debug_information += ["Uniform Exploration"]
            return random.choice(range(self.number_of_actions)) + 1

        self.debug_information += ["Exploit"]
        return self.choose_best_action(history)


class UpperConfidenceBound(Policy):
    def __init__(self, number_of_actions: int, epsilon: float):
        self.epsilon = epsilon
        self.inference = BayesianInference()
        super().__init__(number_of_actions)

    def __str__(self):
        return f"UpperConfidenceBound: {self.epsilon} epsilon"

    def choose_best_action(self, history):
        outcome_groupby = history.to_df().groupby("treatment")["outcome"].mean()
        best_row = outcome_groupby.idxmin()
        return best_row

    def choose_action(self, history, _, block_length):
        self.inference.update_posterior(history, self.number_of_actions)
        bounds = (
            self.inference.get_upper_confidence_bounds("average_treatment_effect")
            .get("average_treatment_effect")
            .data
        )
        upper_bounds_array = [bound[1] for bound in bounds]
        self.debug_information += [
            f"Round {len(history)}: Upper Bounds Array: {upper_bounds_array}"
        ]
        return np.argmax(upper_bounds_array) + 1


class ThompsonSampling(Policy):
    def __init__(self, number_of_actions: int):
        self.inference = BayesianInference()
        super().__init__(number_of_actions)

    def __str__(self):
        return f"ThompsonSampling\n"

    def choose_action(self, history, _, block_length):
        self.inference.update_posterior(history, self.number_of_actions)
        probability_array = self.inference.approximate_max_probabilities()
        action = (
            random.choices(range(self.number_of_actions), weights=probability_array)[0]
            + 1
        )
        self.debug_information += [
            f"Probabilities for picking: {probability_array}, chose {action}"
        ]
        return action
