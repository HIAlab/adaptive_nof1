from src.adaptive_nof1.policies.policy import Policy


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
