from adaptive_nof1.basic_types import History
from adaptive_nof1.models.sinot_model import SinotModel
from adaptive_nof1.policies.block_policy import BlockPolicy
from adaptive_nof1.policies.fixed_policy import FixedPolicy
from adaptive_nof1.simulation import Simulation


def load_sinot_simulation(
    file_path="",
    policy=BlockPolicy(FixedPolicy(number_of_actions=2), block_length=10),
):
    simulation = Simulation(
        **{
            "history": History(observations=[]),
            "model": SinotModel(parameter_file_path=file_path),
            "policy": policy,
        }
    )
    return simulation
