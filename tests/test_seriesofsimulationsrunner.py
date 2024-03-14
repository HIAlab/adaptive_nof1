from adaptive_nof1.basic_types import *

from .fixtures import runner, fixed_policy


def test_clone_with_different_policy(runner, fixed_policy):
    cloned_with_same_policy = runner.clone_with_policy(fixed_policy)

    assert runner.simulate(100) == cloned_with_same_policy.simulate(100)
