import numpy
import pandas as pd
import joblib
import os
from typing import Dict, Any
from adaptive_nof1.basic_types import History
from adaptive_nof1.models.model import Model


class SklearnSimulationModel(Model):
    """
    A simulation model that uses a pre-trained scikit-learn regressor
    to generate context and outcomes.

    This model allows simulation using machine learning models saved
    from previous training sessions.
    """

    def __init__(
        self,
        patient_id: int,
        model_path: str,
        context_features: list = None,
        outcome_features: list = None,
        name: str = "SklearnSimulationModel",
        context_variance: float = 0.1,
        outcome_variance: float = 0.1,
        negate_output: bool = False,
    ):
        """
        Initialize the SklearnSimulationModel.

        :param patient_id: Unique identifier for patient
        :param model_path: Directory path containing saved ML model and preprocessing objects
        :param context_features: Features used for context generation
        :param outcome_features: Features used for outcome generation
        :param name: Name of the model
        :param context_variance: Noise added to context generation
        :param outcome_variance: Noise added to outcome generation
        :param negate_output: If True, negate the predicted outcome
        """
        self.rng = numpy.random.default_rng(patient_id)
        self.patient_id = patient_id
        self.name = name
        self.context_variance = context_variance
        self.outcome_variance = outcome_variance
        self.negate_output = negate_output

        self.model = joblib.load(
            os.path.join(model_path, "survey_response_model.joblib")
        )

        self.context_features = context_features or [
            "weekday",
            "manuscript_ID",
            "intervention_period",
        ]
        self.outcome_features = outcome_features or [
            "weekday",
            "manuscript_ID",
            "intervention_period",
        ]

        self.starting_context = {
            "weekday": "Monday",
            "manuscript_ID": "P_01",
            "intervention_period": "1",
            "current_day_of_study": 0,
            "Y_k_minus_1": 0.0,
            "L": 0,
        }

    def generate_context(self, history: History) -> Dict[str, Any]:
        """
        Generate context based on previous observations.

        :param history: Observation history
        :return: Dictionary of context features
        """
        if not history.observations:
            context = self.starting_context.copy()
        else:
            last_obs = history.observations[-1]
            context = {
                "weekday": last_obs.context.get("weekday", "Monday"),
                "L": (last_obs.context.get("L", 0) + 1) % 7,
                "manuscript_ID": last_obs.context.get("manuscript_ID", "P_01"),
                "intervention_period": last_obs.context.get("intervention_period", "1"),
                "current_day_of_study": last_obs.context.get("current_day_of_study", 0)
                + 1,
                "Y_k_minus_1": last_obs.outcome.get("outcome", 0.0),
            }

        context["design_matrix"] = self._create_design_matrix(
            context.get("L", 0), treatment=0
        )

        return context

    def _create_design_matrix(
        self, weekday_index: int, treatment: int
    ) -> numpy.ndarray:
        """
        Create design matrix with intercept and dummy variables.

        Baseline: Treatment 0 (No Treatment), Monday (weekday_index 0)
        Design matrix: [Intercept, Tuesday-Sunday (6 dummies), Treatment_1]

        :param weekday_index: Day of week index (0=Monday, 1=Tuesday, ..., 6=Sunday)
        :param treatment: Treatment index (0=No Treatment/baseline, 1=Treatment)
        :return: Design matrix as 1D array of shape (8,)
        """
        intercept = 1.0
        treatment_dummy = 1.0 if treatment == 1 else 0.0

        weekday_dummies = numpy.zeros(6)
        if weekday_index > 0:
            weekday_dummies[weekday_index - 1] = 1.0

        return numpy.concatenate([[intercept], weekday_dummies, [treatment_dummy]])

    def observe_outcome(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict outcome using the pre-trained model.

        :param action: Treatment action
        :param context: Context dictionary
        :return: Outcome dictionary
        """
        treatment = action["treatment"]
        weekday_index = context.get("L", 0)
        context["design_matrix"] = self._create_design_matrix(weekday_index, treatment)

        X = pd.DataFrame(
            [[weekday_index, self.patient_id, treatment]],
            columns=["weekday_encoded", "manuscript_encoded", "period_encoded"],
        )

        outcome = self.model.predict(X)[0]
        outcome += self.rng.normal(0, self.outcome_variance)

        if self.negate_output:
            outcome = -outcome

        return {"outcome": outcome}

    @property
    def number_of_interventions(self) -> int:
        """
        Number of possible treatment interventions.

        :return: Number of interventions
        """
        return 2

    @property
    def additional_config(self) -> Dict[str, Any]:
        """
        Additional configuration details.

        :return: Configuration dictionary
        """
        return {
            "model_type": str(type(self.model)),
            "context_variance": self.context_variance,
            "outcome_variance": self.outcome_variance,
            "negate_output": self.negate_output,
        }

    def __str__(self) -> str:
        """
        String representation of the model.

        :return: Model name
        """
        return f"SklearnSimulationModel_{self.name}"
