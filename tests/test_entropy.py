from adaptive_nof1.helpers import *
import pytest

from adaptive_nof1.metrics import Entropy
from .fixtures import *


def test_entropy(simulation_data):
    metric = Entropy(outcome_name="treatment")
    # Values are 1 2 3, so calulation is -3 * 1/3 * ln(1/3)
    assert metric.score(simulation_data) == pytest.approx(1.098612289)

    # Values are 1 1 2, so calulation is -1/3 * ln(1/3) + 2/3 * ln(2/3)
    metric = Entropy(outcome_name="outcome")
    assert metric.score(simulation_data) == pytest.approx(0.6365141683)
