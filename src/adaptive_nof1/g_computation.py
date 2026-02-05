"""
G-computation functions for estimating causal effects with variable covariates.

This module implements g-computation for point estimates and confidence intervals
via parametric bootstrap, with support for dynamic treatments and time-varying covariates.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
import warnings

warnings.filterwarnings("ignore")


def initialize_counterfactual_dataset(dat_train, treatment_strategy):
    """
    Initialize a counterfactual dataset to be filled with simulated outcomes.

    Creates a copy of dat_train where outcome variables are set to NA,
    but with a specified treatment strategy applied throughout. Fixed covariates
    (like weekday) are retained from the original data.

    Parameters
    ----------
    dat_train : pd.DataFrame
        Training data with structure: id, time_discrete, weekday,
        treatment, outcome, and lagged versions of treatment and outcome.
    treatment_strategy : bool, array-like, or pd.Series
        Treatment strategy to apply. Can be:
        - Single bool: apply constant treatment (True/False) to all observations
        - Array-like or pd.Series: apply specific treatment at each time point

    Returns
    -------
    pd.DataFrame
        Initialized counterfactual dataset with treatment strategy applied.
    """
    # Create a copy of the training data
    temp_d = dat_train.copy()

    # Variables to set to NA (will be filled during simulation)
    vars_to_na = ["lag1_treatment", "lag1_outcome", "treatment", "outcome"]

    # Set specified variables to NA
    for var in vars_to_na:
        if var in temp_d.columns:
            temp_d[var] = np.nan

    # Apply treatment strategy
    if isinstance(treatment_strategy, (bool, np.bool_)):
        temp_d["treatment"] = treatment_strategy
    else:
        temp_d["treatment"] = treatment_strategy

    # Fix first time point - use observed values from dat_train
    first_point_vars = ["treatment", "outcome"]
    temp_d.loc[0, first_point_vars] = dat_train.loc[0, first_point_vars].values

    # Set lagged values for second observation
    if len(temp_d) > 1:
        temp_d.loc[1, "lag1_treatment"] = dat_train.loc[0, "treatment"]
        temp_d.loc[1, "lag1_outcome"] = dat_train.loc[0, "outcome"]

    return temp_d

def fill_counterfactual_dataset(temp_d, mod_out):
    """
    Fill counterfactual dataset with simulated outcomes.

    Simulates outcomes based on the fitted outcome model. Covariates (weekday)
    are fixed and carried forward from the original data.

    Parameters
    ----------
    temp_d : pd.DataFrame
        Partially initialized counterfactual dataset from initialize_counterfactual_dataset()
    mod_out : statsmodels model
        Fitted outcome model for prediction

    Returns
    -------
    pd.DataFrame
        Counterfactual dataset with simulated outcomes
    """
    temp_d = temp_d.copy()

    for i in range(len(temp_d)):
        # Simulate outcome using the outcome model
        pred_outcome = mod_out.predict(temp_d.loc[[i], :])
        pred_prob = pred_outcome.values[0]

        # Extract precision parameter (phi) from model
        if hasattr(mod_out, "params_precision"):
            precision = float(mod_out.params_precision)
        elif "precision" in mod_out.params.index:
            precision = float(mod_out.params["precision"])
        else:
            # Default if precision not found
            precision = 10.0

        # Calculate beta shape parameters
        shape1 = pred_prob * precision
        shape2 = precision - shape1
        temp_d.loc[i, "outcome"] = np.random.beta(shape1, shape2)

        # Update lagged values for next iteration
        if i + 1 < len(temp_d):
            temp_d.loc[i + 1, "lag1_treatment"] = temp_d.loc[i, "treatment"]
            temp_d.loc[i + 1, "lag1_outcome"] = temp_d.loc[i, "outcome"]

    return temp_d


def prepare_training_data(dat_i):
    """
    Prepare training data with lagged variables for g-computation.

    Takes raw data and creates lagged versions of treatment and outcome
    for modeling. Covariates like weekday are preserved as-is.

    Parameters
    ----------
    dat_i : pd.DataFrame
        Raw data with columns: id, time_discrete, weekday, treatment, outcome

    Returns
    -------
    pd.DataFrame
        Data with lagged treatment and outcome added as lag1_* columns
    """
    # Select and prepare data
    dat_train = dat_i[["id", "time_discrete", "weekday", "treatment", "outcome"]].copy()

    # Create lagged versions - shift by 1 and prepend NA row
    lag1_data = dat_train[["treatment", "outcome"]].copy()
    lag1_data = pd.concat(
        [pd.DataFrame([[np.nan, np.nan]], columns=["treatment", "outcome"]), lag1_data]
    ).reset_index(drop=True)
    lag1_data = lag1_data.iloc[:-1].reset_index(drop=True)

    # Rename columns for lagging
    lag1_data.columns = ["lag1_treatment", "lag1_outcome"]

    # Combine original and lagged data
    dat_train = dat_train.reset_index(drop=True)
    dat_train = pd.concat([lag1_data, dat_train], axis=1)

    return dat_train


def fit_outcome_model(dat_train, formula="outcome ~ treatment + weekday + lag1_treatment + lag1_outcome"):
    """
    Fit beta regression model for outcome prediction.

    Parameters
    ----------
    dat_train : pd.DataFrame
        Training data with all variables
    formula : str, optional
        Formula for the model using R-style syntax

    Returns
    -------
    statsmodels model fit result
        Fitted outcome model
    """
    # For beta regression, we use a workaround with statsmodels
    # Create a GLM with binomial family as a practical approximation for proportions
    try:
        from statsmodels.genmod.families import Binomial

        model = smf.gee(formula, groups="id", data=dat_train, family=Binomial())
        result = model.fit()
    except ImportError:
        # Fallback: use OLS with logit transformation
        y = dat_train["outcome"]
        y = np.clip(y, 0.001, 0.999)  # Avoid 0 and 1
        y_logit = np.log(y / (1 - y))

        X = smf.dmatrix(formula.split("~")[1], data=dat_train, return_type="dataframe")
        X = sm.add_constant(X)

        result = sm.OLS(y_logit, X).fit()

    return result

def compute_point_effect(dat_train, model_outcome, G=100):
    """
    Compute point estimates of causal effect via g-computation.

    Estimates the treatment effect by comparing outcomes under "always treat"
    vs "never treat" strategies, averaged over G simulated counterfactual trajectories.

    Parameters
    ----------
    dat_train : pd.DataFrame
        Training data with lagged variables
    model_outcome : fitted model
        Fitted outcome model
    G : int, optional
        Number of simulations for averaging (default: 100)

    Returns
    -------
    np.ndarray
        Point estimates of treatment effect at each time point
    """
    point_effect = np.full((G, len(dat_train)), np.nan)

    for g in range(G):
        # Counterfactual under "never treat" (treatment = False)
        dat_cf_0 = initialize_counterfactual_dataset(dat_train, treatment_strategy=False)
        dat_cf_0 = fill_counterfactual_dataset(dat_cf_0, model_outcome)
        theta_0 = dat_cf_0["outcome"].values

        # Counterfactual under "always treat" (treatment = True)
        dat_cf_1 = initialize_counterfactual_dataset(dat_train, treatment_strategy=True)
        dat_cf_1 = fill_counterfactual_dataset(dat_cf_1, model_outcome)
        theta_1 = dat_cf_1["outcome"].values

        # Treatment effect as difference
        point_effect[g, :] = theta_1 - theta_0

    # Average across simulations
    point_effect_avg = np.mean(point_effect, axis=0)

    return point_effect_avg


def parametric_bootstrap_confidence_intervals(
    dat_train, model_outcome, point_estimate, B=100, G=50, alpha=0.05
):
    """
    Compute confidence intervals via parametric bootstrap.

    Fits models to bootstrap samples of generated counterfactual data and
    computes the variability of effect estimates.

    Parameters
    ----------
    dat_train : pd.DataFrame
        Training data with lagged variables
    model_outcome : fitted model
        Fitted outcome model from original data
    point_estimate : np.ndarray
        Point estimates of treatment effect
    B : int, optional
        Number of bootstrap samples (default: 100)
    G : int, optional
        Number of simulations per bootstrap sample (default: 50)
    alpha : float, optional
        Significance level for confidence intervals (default: 0.05)

    Returns
    -------
    pd.DataFrame
        Report with point estimates, standard errors, and confidence intervals
    """
    effects = np.full((B, len(dat_train)), np.nan)

    for b in range(B):
        # Generate bootstrap sample by simulating from model under observed treatment
        dat_boot = initialize_counterfactual_dataset(dat_train, treatment_strategy=dat_train["treatment"].values)
        dat_boot = fill_counterfactual_dataset(dat_boot, model_outcome)

        # Fit outcome model on bootstrap sample
        try:
            model_out_boot = fit_outcome_model(dat_boot)

            # Compute point effect in bootstrap sample
            point_effect_boot = compute_point_effect(dat_train, model_out_boot, G=G)
            effects[b, :] = point_effect_boot
        except Exception as e:
            print(f"Bootstrap iteration {b+1} failed: {e}")
            effects[b, :] = point_estimate

    # Compute standard errors and confidence intervals
    std_errors = np.std(effects, axis=0)

    z_critical = scipy_norm.ppf(1 - alpha / 2)

    report = pd.DataFrame(
        {
            "point": point_estimate,
            "std": std_errors,
            "lower": point_estimate - z_critical * std_errors,
            "upper": point_estimate + z_critical * std_errors,
            "time_discrete": np.arange(1, len(dat_train) + 1),
        }
    )

    return report


def run_g_computation(
    dat_i,
    G=100,
    B=100,
    alpha=0.05,
    outcome_formula="outcome ~ treatment + weekday + lag1_treatment + lag1_outcome",
):
    """
    Run complete g-computation pipeline for causal effect estimation.

    Orchestrates data preparation, model fitting, point estimation, and
    confidence interval computation.

    Parameters
    ----------
    dat_i : pd.DataFrame
        Raw input data with columns: id, time_discrete, weekday,
        treatment, outcome
    G : int, optional
        Number of simulations for point estimation (default: 100)
    B : int, optional
        Number of bootstrap samples (default: 100)
    alpha : float, optional
        Significance level for confidence intervals (default: 0.05)
    outcome_formula : str, optional
        Formula for outcome model

    Returns
    -------
    pd.DataFrame
        Report with point estimates, standard errors, and confidence intervals
    """
    # Prepare data with lagged variables
    dat_train = prepare_training_data(dat_i)

    # Fit outcome model on original data
    model_outcome = fit_outcome_model(dat_train, formula=outcome_formula)

    # Compute point estimates
    point_effect = compute_point_effect(dat_train, model_outcome, G=G)

    # Compute confidence intervals via parametric bootstrap
    report = parametric_bootstrap_confidence_intervals(
        dat_train, model_outcome, point_effect, B=B, G=G, alpha=alpha
    )

    return report
