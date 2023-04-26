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
        best_row = outcome_groupby.idxmin()
        return best_row

    def choose_action(self, history, _):
        if self.best_treatment:
            return self.best_treatment
        if len(history) >= self.explore_blocks * self.fixed_policy.block_length:
            self.best_treatment = self.choose_best_action(history)
            return self.best_treatment
        return self.fixed_policy.choose_action(history, _)
