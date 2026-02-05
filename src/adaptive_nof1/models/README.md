# Custom Adaptive N-of-1 Model Implementations

## SklearnSimulationModel

The `SklearnSimulationModel` is a flexible simulation model that can use pre-trained scikit-learn regressors for generating context and outcomes in adaptive trials.

### Key Features
- Loads pre-trained machine learning models
- Supports label encoding and feature scaling
- Adds configurable noise to context and outcomes
- Compatible with the adaptive N-of-1 trial framework

### Usage Example

```python
from adaptive_nof1.models.sklearn_simulation_model import SklearnSimulationModel

# Initialize the model with a path to saved model artifacts
model = SklearnSimulationModel(
    patient_id=42, 
    model_path='/path/to/saved/model/directory',
    name='survey_response_simulator'
)

# Use in an adaptive trial simulation
history = History()  # From adaptive_nof1.basic_types
context = model.generate_context(history)
action = {'treatment': 1}  # Example treatment action
outcome = model.observe_outcome(action, context)
```

### Model Directory Structure
Your model directory should contain:
- `survey_response_model.joblib`: Trained sklearn regressor
- `le_weekday.joblib`: Weekday label encoder
- `le_manuscript.joblib`: Manuscript ID label encoder
- `le_period.joblib`: Study period label encoder
- `feature_scaler.joblib`: Feature scaler (StandardScaler)
- `README.md`: Optional model description

### Configuration Parameters
- `patient_id`: Random seed for reproducibility
- `model_path`: Directory with saved model artifacts
- `context_features`: Optional custom context features
- `outcome_features`: Optional custom outcome features
- `context_variance`: Noise level for context generation
- `outcome_variance`: Noise level for outcome prediction

### Limitations
- Requires pre-trained scikit-learn model
- Assumes specific preprocessing (label encoding, scaling)
- May not capture complex non-linear relationships