from adaptive_nof1.policies.policy import Policy


class FixedPolicy(Policy):
    def __str__(self):
        return f"FixedPolicy\n"

    def choose_action(self, history, _, block_length=None):
        block_length = 1 if block_length is None else block_length
        round = len(history) // block_length
        self._debug_information += ["Fixed Schedule"]
        return {self.treatment_name: round % self.number_of_actions + 1}
