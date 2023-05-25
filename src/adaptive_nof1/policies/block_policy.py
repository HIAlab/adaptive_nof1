from adaptive_nof1.policies.policy import Policy


class BlockPolicy(Policy):
    def __init__(self, internal_policy: Policy, block_length):
        self.internal_policy = internal_policy
        self.block_length = block_length
        self.last_action = None
        super().__init__(number_of_actions=internal_policy.number_of_actions)

    def __str__(self):
        return f"BlockPolicy({self.internal_policy})"

    def is_first_of_block(self, number):
        return number % self.block_length == 0

    def choose_action(self, history, context):
        if self.is_first_of_block(len(history)):
            self.last_action = self.internal_policy.choose_action(
                history, context, block_length=self.block_length
            )
            self._debug_information += [self.internal_policy.debug_information[-1]]
        else:
            self._debug_information += ["Repeated Block"]
        return self.last_action
