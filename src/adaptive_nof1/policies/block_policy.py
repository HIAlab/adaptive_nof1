from adaptive_nof1.policies.policy import Policy


class BlockPolicy(Policy):
    def __init__(self, internal_policy: Policy, block_length, **kwargs):
        self.internal_policy = internal_policy
        self.block_length = block_length
        self.last_action = None
        super().__init__(**kwargs)

    def __str__(self):
        return f"BlockPolicy({self.internal_policy})"

    @property
    def number_of_actions(self):
        return self.internal_policy.number_of_actions

    @number_of_actions.setter
    def number_of_actions(self, value):
        self.internal_policy.number_of_actions = value

    @property
    def additional_config(self):
        return self.internal_policy.additional_config

    @property
    def is_stopped(self):
        return self.internal_policy.is_stopped

    def is_first_of_block(self, number):
        return number % self.block_length == 0

    def choose_action(self, history, context):
        if self.is_first_of_block(context["t"]):
            self.last_action = self.internal_policy.choose_action(
                history,
                context,
            )
            self._debug_information += [self.internal_policy.debug_information[-1]]
            self._debug_data += [
                {**self.internal_policy.debug_data[-1], "is_start_of_block": True}
            ]
        else:
            self._debug_information += ["Repeated Block"]
            self._debug_data += [
                {"is_start_of_block": False, **self.internal_policy.debug_data[-1]}
            ]
        return self.last_action
