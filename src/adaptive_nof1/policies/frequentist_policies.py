from src.adaptive_nof1.policies.fixed_policy import FixedPolicy
from src.adaptive_nof1.policies.policy import Policy


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
