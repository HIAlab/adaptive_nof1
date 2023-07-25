import pytest
from adaptive_nof1 import SeriesOfSimulationsRunner
from adaptive_nof1.models.pill_model import PillModel
from adaptive_nof1.policies import CombinedPolicy, FixedPolicy

NUMBER_OF_ACTIONS = 5
NUMBER_OF_PATIENTS = 5


@pytest.fixture
def fixed_policy():
    return CombinedPolicy(
        [
            FixedPolicy(number_of_actions=NUMBER_OF_ACTIONS),
        ],
        number_of_actions=NUMBER_OF_ACTIONS,
    )


@pytest.fixture
def runner(fixed_policy):
    runner = SeriesOfSimulationsRunner(
        model_from_patient_id=lambda patient_id: PillModel(patient_id=patient_id),
        n_patients=NUMBER_OF_PATIENTS,
        policy=fixed_policy,
    )
    return runner
