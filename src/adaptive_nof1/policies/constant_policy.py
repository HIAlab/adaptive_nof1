from adaptive_nof1.policies.policy import Policy


class ConstantPolicy(Policy):
    def __init__(self, number_of_actions, action):
        self.action = action
        super().__init__(number_of_actions)

    def __str__(self):
        return f"ConstantPolicy: {self.action}.\n"

    def choose_action(self):
        return self.action
