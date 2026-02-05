#!/usr/bin/env python
"""
Grid Search Script for SCM Model 06 (Fu et al. Acne Trial Simulation)

This script performs a grid search over:
- BETA_RESPONDER: Treatment effect for responders
- phi: AR(1) autocorrelation coefficient
- ell: Effect of temperature covariate on outcome

Metrics computed:
1. BAI (Best Arm Identification): Accuracy of identifying the best treatment
2. Time of optimal treatment: Fraction of periods receiving the best treatment
3. Bias: Bias in estimating the treatment effect
"""

import argparse
import itertools
import json
import numpy as np
import pandas as pd
import sklearn.linear_model
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from adaptive_nof1 import simulate_configurations
from adaptive_nof1.helpers import generate_configuration_cross_product, write_to_disk
from adaptive_nof1.policies import (
    BlockPolicy,
    FixedPolicy,
    ConstantPolicy,
    ThompsonSampling,
)
from adaptive_nof1.inference import BayesianLinearRegressionModel
from adaptive_nof1.models import Model


# =============================================================================
# Model Definition
# =============================================================================


class FuTrialSCM(Model):
    """
    Structural Causal Model for Fu et al. acne trial simulation.

    Covariate model (AR(1) for temperature):
        γ_k ~ N(a, b)
        L_k = φ * L_{k-1} + γ_k

    Outcome model:
        Y_k = μ + β * A_k + ℓ * L_k + ε
    """

    def __init__(
        self,
        patient_id,
        beta,
        mu=0.5,
        ell=0.05,
        phi=0.3,
        a=0.0,
        b=1.0,
        outcome_variance=1,
        name="FuTrialSCM",
    ):
        self.patient_id = patient_id
        self.rng = np.random.default_rng(patient_id)

        self.beta = beta
        self.mu = mu
        self.ell = ell
        self.phi = phi
        self.a = a
        self.b = b
        self.outcome_variance = outcome_variance
        self.name = name

    def generate_context(self, history):
        """Generate temperature covariate using AR(1) model."""
        if len(history.observations) == 0:
            L = self.rng.normal(0, self.b / np.sqrt(1 - self.phi**2))
            return {"L": L, "Y_k_minus_1": 0.0}

        last_obs = history.observations[-1]
        L_k_minus_1 = last_obs.context["L"]
        Y_k_minus_1 = last_obs.outcome["outcome"]

        gamma_k = self.rng.normal(self.a, self.b)
        L_k = self.phi * L_k_minus_1 + gamma_k

        return {"L": L_k, "Y_k_minus_1": Y_k_minus_1}

    @property
    def number_of_interventions(self):
        return 2

    @property
    def additional_config(self):
        return {
            "beta": self.beta,
            "mu": self.mu,
            "ell": self.ell,
            "phi": self.phi,
            "is_responder": self.beta != 0,
        }

    def observe_outcome(self, action, context):
        """Generate outcome: Y_k = μ + β * A_k + ℓ * L_k + ε"""
        A_k = action["treatment"]
        L_k = context["L"]

        epsilon = self.rng.normal(0, np.sqrt(self.outcome_variance))
        Y_k = self.mu + self.beta * A_k + self.ell * L_k + epsilon

        return {"outcome": Y_k}

    def __str__(self):
        responder_str = "responder" if self.beta != 0 else "non_responder"
        return f"{self.name}_{responder_str}"


# =============================================================================
# Simulation Setup
# =============================================================================

# Prior specification for Bayesian inference
PRIOR_SPEC = {
    "cov_diag": 10.0,
    "alpha": 2.0,
    "beta": 2.0,
}

# Trial configuration (following Fu et al.)
DAYS_PER_PERIOD = 2
NUMBER_OF_PERIODS = 8
BLOCK_LENGTH = DAYS_PER_PERIOD
SIMULATION_LENGTH = NUMBER_OF_PERIODS * BLOCK_LENGTH


def create_inference_model(include_covariate=True):
    """Create Bayesian linear regression model for inference."""
    if include_covariate:
        mean = np.array([0.0, 0.0, 0.0])
        v = np.eye(3) * PRIOR_SPEC["cov_diag"]
        covariate_names = ["L"]
        name = "With_Covariate"
    else:
        mean = np.array([0.0, 0.0])
        v = np.eye(2) * PRIOR_SPEC["cov_diag"]
        covariate_names = []
        name = "Without_Covariate"

    return BayesianLinearRegressionModel(
        mean=mean,
        v=v,
        alpha=PRIOR_SPEC["alpha"],
        beta=PRIOR_SPEC["beta"],
        covariate_names=covariate_names,
        name=name,
    )


def create_policies():
    """Create the set of policies to compare."""
    policies = [
        BlockPolicy(
            block_length=BLOCK_LENGTH,
            internal_policy=FixedPolicy(
                inference_model=create_inference_model(include_covariate=False),
                block_length=BLOCK_LENGTH,
                randomize=True,
            ),
        ),
        BlockPolicy(
            block_length=BLOCK_LENGTH,
            internal_policy=ConstantPolicy(action=0),
        ),
        BlockPolicy(
            block_length=BLOCK_LENGTH,
            internal_policy=ConstantPolicy(action=1),
        ),
        BlockPolicy(
            block_length=BLOCK_LENGTH,
            internal_policy=ThompsonSampling(
                inference_model=create_inference_model(include_covariate=False),
            ),
        ),
        BlockPolicy(
            block_length=BLOCK_LENGTH,
            internal_policy=ThompsonSampling(
                inference_model=create_inference_model(include_covariate=True),
            ),
        ),
    ]

    policy_names = [
        "Fixed Randomization",
        "Never Treat",
        "Always Treat",
        "Thompson Sampling (no cov)",
        "Thompson Sampling (with cov)",
    ]

    return policies, policy_names


# =============================================================================
# Metric Computation
# =============================================================================


def compute_metrics(simulation_results, beta_responder, policy_mapping):
    """
    Compute all three metrics:
    1. BAI (Best Arm Identification accuracy)
    2. Time of optimal treatment (fraction of periods)
    3. Bias (in treatment effect estimation)

    Returns a DataFrame with all metrics.
    """
    results = []

    for sim in simulation_results:
        config = sim["configuration"]
        result = sim["result"]

        model_str = str(config.get("model", ""))
        is_responder = (
            "responder" in model_str.lower() and "non" not in model_str.lower()
        )
        true_beta = beta_responder if is_responder else 0.0

        policy_str = str(config.get("policy", ""))
        policy_name = policy_mapping.get(policy_str, policy_str)

        for patient_data in result.simulations:
            # Prepare data for inference
            X_list = []
            y_list = []
            treatment_count = 0
            total_periods = 0

            for obs in patient_data.history.observations:
                action = obs.treatment["treatment"]
                L_k = obs.context["L"]
                outcome = obs.outcome["outcome"]

                X_list.append([action, L_k])
                y_list.append(outcome)

                total_periods += 1
                if action == 1:
                    treatment_count += 1

            if len(X_list) > 0:
                X = np.array(X_list)
                y = np.array(y_list)

                # Fit Bayesian Ridge regression
                bayesian_model = sklearn.linear_model.BayesianRidge(
                    alpha_1=PRIOR_SPEC["alpha"],
                    alpha_2=PRIOR_SPEC["beta"],
                    lambda_1=PRIOR_SPEC["alpha"],
                    lambda_2=PRIOR_SPEC["beta"],
                    compute_score=True,
                )
                bayesian_model.fit(X, y)

                estimated_beta = bayesian_model.coef_[0]

                # BAI: treatment is best if β > 0
                identified_best_arm = 1 if estimated_beta > 0 else 0
                true_best_arm = 1 if is_responder else 0
                correct_identification = (
                    (identified_best_arm == true_best_arm) if is_responder else None
                )

                # Bias
                bias = estimated_beta - true_beta

                # Treatment allocation
                best_treatment_fraction = (
                    treatment_count / total_periods if total_periods > 0 else 0
                )

                results.append(
                    {
                        "policy": policy_name,
                        "is_responder": is_responder,
                        "bai_correct": correct_identification,
                        "optimal_treatment_fraction": best_treatment_fraction,
                        "estimated_beta": estimated_beta,
                        "true_beta": true_beta,
                        "bias": bias,
                        "absolute_error": abs(bias),
                    }
                )

    return pd.DataFrame(results)


def aggregate_metrics(metrics_df):
    """
    Aggregate metrics across patients for each policy.

    Returns summary statistics for each metric.
    """
    # Split by responder status
    responder_df = metrics_df[metrics_df["is_responder"] == True]
    non_responder_df = metrics_df[metrics_df["is_responder"] == False]

    results = {}

    # Responder metrics
    if len(responder_df) > 0:
        responder_agg = responder_df.groupby("policy").agg(
            {
                "bai_correct": ["mean", "std", "count"],
                "optimal_treatment_fraction": ["mean", "std"],
                "bias": ["mean", "std"],
                "absolute_error": ["mean", "std"],
            }
        )
        responder_agg.columns = [
            "_".join(col).strip() for col in responder_agg.columns.values
        ]
        results["responder"] = responder_agg.reset_index().to_dict("records")

    # Non-responder metrics (bias only, BAI is undefined)
    if len(non_responder_df) > 0:
        non_responder_agg = non_responder_df.groupby("policy").agg(
            {
                "optimal_treatment_fraction": ["mean", "std"],
                "bias": ["mean", "std"],
                "absolute_error": ["mean", "std"],
            }
        )
        non_responder_agg.columns = [
            "_".join(col).strip() for col in non_responder_agg.columns.values
        ]
        results["non_responder"] = non_responder_agg.reset_index().to_dict("records")

    return results


# =============================================================================
# Grid Search
# =============================================================================


def create_model_generator(beta, phi, ell, a, b, is_responder):
    """Create a model generator function for given parameters."""
    name = f"FuTrial_{'Responder' if is_responder else 'NonResponder'}"

    def generator(patient_id):
        return FuTrialSCM(
            patient_id=patient_id,
            beta=beta if is_responder else 0.0,
            phi=phi,
            a=a,
            b=b,
            ell=ell,
            name=name,
        )

    return generator


def run_single_simulation(beta_responder, phi, ell, n_patients, policies, policy_names):
    """
    Run a single simulation with given parameters.

    Returns the computed metrics DataFrame.
    """
    # Default AR(1) parameters (fitted from temperature data in notebook)
    a = -0.0103
    b = 1.9305

    # Create generators
    responder_generator = create_model_generator(
        beta=beta_responder, phi=phi, ell=ell, a=a, b=b, is_responder=True
    )
    non_responder_generator = create_model_generator(
        beta=0.0, phi=phi, ell=ell, a=a, b=b, is_responder=False
    )

    # Create configurations
    study_designs = {
        "n_patients": [n_patients],
        "policy": policies,
        "model_from_patient_id": [responder_generator, non_responder_generator],
    }

    configurations = generate_configuration_cross_product(study_designs)

    # Run simulation
    calculated_series, _ = simulate_configurations(configurations, SIMULATION_LENGTH)

    # Create policy mapping
    policy_mapping = {str(p): name for p, name in zip(policies, policy_names)}

    # Compute metrics
    metrics_df = compute_metrics(calculated_series, beta_responder, policy_mapping)

    return metrics_df


def run_grid_search(
    beta_values,
    phi_values,
    ell_values,
    n_patients=100,
    output_dir="results",
):
    """
    Run grid search over all parameter combinations.

    Args:
        beta_values: List of BETA_RESPONDER values
        phi_values: List of phi (autocorrelation) values
        ell_values: List of ell (covariate effect) values
        n_patients: Number of patients per simulation
        output_dir: Directory to save results

    Returns:
        DataFrame with all results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create policies once
    policies, policy_names = create_policies()

    # Generate all parameter combinations
    param_combinations = list(itertools.product(beta_values, phi_values, ell_values))

    print(
        f"Running grid search over {len(param_combinations)} parameter combinations..."
    )
    print(f"  BETA_RESPONDER: {beta_values}")
    print(f"  phi (autocorrelation): {phi_values}")
    print(f"  ell (covariate effect): {ell_values}")
    print(f"  n_patients: {n_patients}")
    print()

    all_results = []

    for beta, phi, ell in tqdm(param_combinations, desc="Grid search"):
        try:
            metrics_df = run_single_simulation(
                beta_responder=beta,
                phi=phi,
                ell=ell,
                n_patients=n_patients,
                policies=policies,
                policy_names=policy_names,
            )

            # Add parameter columns
            metrics_df["param_beta"] = beta
            metrics_df["param_phi"] = phi
            metrics_df["param_ell"] = ell

            all_results.append(metrics_df)

        except Exception as e:
            print(f"Error for beta={beta}, phi={phi}, ell={ell}: {e}")
            continue

    if not all_results:
        print("No results generated!")
        return None

    # Combine all results
    full_results = pd.concat(all_results, ignore_index=True)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_path = output_path / f"grid_search_detailed_{timestamp}.csv"
    full_results.to_csv(detailed_path, index=False)
    print(f"\nDetailed results saved to: {detailed_path}")

    # Create and save summary
    summary = create_summary(full_results)
    summary_path = output_path / f"grid_search_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary results saved to: {summary_path}")

    # Save parameters
    params = {
        "beta_values": beta_values,
        "phi_values": phi_values,
        "ell_values": ell_values,
        "n_patients": n_patients,
        "timestamp": timestamp,
    }
    params_path = output_path / f"grid_search_params_{timestamp}.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Parameters saved to: {params_path}")

    return full_results


def create_summary(full_results):
    """
    Create a summary DataFrame aggregating metrics by parameter combination and policy.
    """
    # Group by parameters and policy
    summary_list = []

    for (beta, phi, ell, policy), group in full_results.groupby(
        ["param_beta", "param_phi", "param_ell", "policy"]
    ):
        # Responder metrics
        responder = group[group["is_responder"] == True]
        non_responder = group[group["is_responder"] == False]

        row = {
            "param_beta": beta,
            "param_phi": phi,
            "param_ell": ell,
            "policy": policy,
        }

        if len(responder) > 0:
            row["bai_accuracy"] = responder["bai_correct"].mean()
            row["bai_accuracy_std"] = responder["bai_correct"].std()
            row["optimal_treatment_fraction_responder"] = responder[
                "optimal_treatment_fraction"
            ].mean()
            row["optimal_treatment_fraction_responder_std"] = responder[
                "optimal_treatment_fraction"
            ].std()
            row["bias_responder"] = responder["bias"].mean()
            row["bias_responder_std"] = responder["bias"].std()
            row["mae_responder"] = responder["absolute_error"].mean()

        if len(non_responder) > 0:
            row["optimal_treatment_fraction_non_responder"] = non_responder[
                "optimal_treatment_fraction"
            ].mean()
            row["bias_non_responder"] = non_responder["bias"].mean()
            row["mae_non_responder"] = non_responder["absolute_error"].mean()

        summary_list.append(row)

    return pd.DataFrame(summary_list)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for SCM Model 06 (Fu et al. Acne Trial)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--beta",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4],
        help="BETA_RESPONDER values (treatment effect for responders)",
    )
    parser.add_argument(
        "--phi",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8],
        help="phi values (AR(1) autocorrelation coefficient)",
    )
    parser.add_argument(
        "--ell",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2],
        help="ell values (effect of temperature covariate on outcome)",
    )
    parser.add_argument(
        "--n-patients", type=int, default=100, help="Number of patients per simulation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/grid_search",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Run grid search
    results = run_grid_search(
        beta_values=args.beta,
        phi_values=args.phi,
        ell_values=args.ell,
        n_patients=args.n_patients,
        output_dir=args.output_dir,
    )

    if results is not None:
        print("\nGrid search complete!")
        print(f"Total simulations: {len(results)}")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY BY PARAMETER")
        print("=" * 80)

        # Summary by beta
        print("\nBAI Accuracy by BETA_RESPONDER (responders only):")
        responder_results = results[results["is_responder"] == True]
        for beta in args.beta:
            subset = responder_results[responder_results["param_beta"] == beta]
            if len(subset) > 0:
                acc = subset["bai_correct"].mean()
                print(f"  β = {beta}: {acc:.1%}")

        # Summary by phi
        print("\nBAI Accuracy by phi (responders only):")
        for phi in args.phi:
            subset = responder_results[responder_results["param_phi"] == phi]
            if len(subset) > 0:
                acc = subset["bai_correct"].mean()
                print(f"  φ = {phi}: {acc:.1%}")

        # Summary by ell
        print("\nBAI Accuracy by ell (responders only):")
        for ell in args.ell:
            subset = responder_results[responder_results["param_ell"] == ell]
            if len(subset) > 0:
                acc = subset["bai_correct"].mean()
                print(f"  ℓ = {ell}: {acc:.1%}")


if __name__ == "__main__":
    main()
