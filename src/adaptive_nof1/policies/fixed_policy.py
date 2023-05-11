from adaptive_nof1.policies.policy import Policy


class FixedPolicy(Policy):
    def __init__(self, number_of_actions):
        super().__init__(number_of_actions)

    def __str__(self):
        return f"FixedPolicy\n"

    def choose_action(self, history, _, block_length=1):
        round = len(history) // block_length
        return round % self.number_of_actions + 1
