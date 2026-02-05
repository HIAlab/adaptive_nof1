import numpy

import pandas
import scipy
import pandas as pd


## Code from: https://github.com/tonyduan/conjugate-bayes/blob/master/conjugate_bayes/models.py
class NIGLinearRegression(object):
    """
    The normal inverse-gamma prior for a linear regression model with unknown
    variance and unknown relationship. Specifically,
        1/σ² ~ Γ(a, b)
        β ~ N(0, σ²V)

    Parameters
    ----------
    mu: prior for N(mu, v) on the model β
    v:  prior for N(mu, v) on the model β
    a:  prior for Γ(a, b) on the inverse sigma2 of the distribution
    b:  prior for Γ(a, b) on the inverse sigma2 of the distribution
    """

    def __init__(self, mu, v, a, b):
        self.__dict__.update({"mu": mu, "v": v, "a": a, "b": b})

    def fit(self, x_tr, y_tr):
        m, _ = x_tr.shape
        mu_ast = numpy.linalg.inv(numpy.linalg.inv(self.v) + x_tr.T @ x_tr) @ (
            numpy.linalg.inv(self.v) @ self.mu + x_tr.T @ y_tr
        )
        v_ast = numpy.linalg.inv(numpy.linalg.inv(self.v) + x_tr.T @ x_tr)
        a_ast = self.a + 0.5 * m
        b_ast = self.b + 0.5 * (y_tr - x_tr @ self.mu).T @ numpy.linalg.inv(
            numpy.eye(m) + x_tr @ self.v @ x_tr.T
        ) @ (y_tr - x_tr @ self.mu.T)
        self.__dict__.update({"mu": mu_ast, "v": v_ast, "a": a_ast, "b": b_ast})

    def predict(self, x_te):
        scales = numpy.array([x.T @ self.v @ x for x in x_te]) + 1
        scales = (self.b / self.a * scales) ** 0.5
        return scipy.stats.t(df=2 * self.a, loc=x_te @ self.mu, scale=scales)

    def get_conditional_beta(self, sigma2):
        return scipy.stats.multivariate_normal(mean=self.mu, cov=sigma2 * self.v)

    def get_marginal_sigma2(self):
        return scipy.stats.invgamma(self.a, scale=self.b)


class BayesianLinearRegressionModel:
    def __init__(
        self,
        treatment_name="treatment",
        outcome_name="outcome",
        mean=0,
        v=0,
        alpha=0,
        beta=0,
        covariate_names=None,
        categorical_covariates=None,
        name=None,
    ):
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self.covariate_names = covariate_names or []
        self.categorical_covariates = categorical_covariates or {}
        self.name = name

        # Ensure mean is a numpy array
        self.mean = numpy.asarray(mean) if not isinstance(mean, numpy.ndarray) else mean
        self.v = v
        self.alpha = alpha
        self.beta = beta

        self.df = None
        self.model = NIGLinearRegression(
            mu=self.mean, v=self.v, a=self.alpha, b=self.beta
        )

        # Validate categorical covariates
        for col, categories in self.categorical_covariates.items():
            if col not in self.covariate_names:
                self.covariate_names.append(col)

    def _prepare_design_matrix(self, df, number_of_treatments):
        """
        Prepare design matrix with treatment dummies and covariates, including one-hot encoded categorical covariates.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe containing treatment and outcome columns
        number_of_treatments : int
            Number of treatments

        Returns
        -------
        numpy.ndarray
            Design matrix with treatment dummies and covariates
        """
        # Get treatment dummies
        treatment_dummies = (
            pandas.get_dummies(
                pandas.Categorical(
                    df[self.treatment_name], categories=range(number_of_treatments)
                )
            )
            .to_numpy()
            .astype(float)
        )

        # Prepare covariates matrix
        covariates_list = [treatment_dummies]

        # Handle categorical covariates with one-hot encoding
        for col, categories in self.categorical_covariates.items():
            # Create categorical with predefined categories to ensure consistency
            cat_series = pandas.Categorical(df[col], categories=range(categories))
            # Get dummies, converting to numpy
            col_dummies = pandas.get_dummies(cat_series).to_numpy().astype(float)
            covariates_list.append(col_dummies)

        # Add continuous covariates
        continuous_covariates_cols = [
            col
            for col in self.covariate_names
            if col not in self.categorical_covariates
        ]
        if continuous_covariates_cols:
            continuous_data = df[continuous_covariates_cols].to_numpy().astype(float)
            covariates_list.append(continuous_data)

        # Combine all covariates
        design_matrix = numpy.hstack(covariates_list)

        return design_matrix

    def get_upper_confidence_bounds(self, variable_name, epsilon: float = 0.05):
        raise AssertionError("Not implemented")

    def update_posterior(self, history, number_of_treatments):
        self.history = history
        if len(history.observations) == 0:
            return
        self.model.fit(
            self._prepare_design_matrix(self.history.to_df(), number_of_treatments),
            self.history.outcomes(),
        )

    def series(self, intervention):
        return self.df[self.df[self.treatment_name] == intervention][self.outcome_name]

    def posterior(self, intervention):
        identity_matrix = numpy.eye(len(self.mean))
        return self.model.predict([identity_matrix[intervention]])

    def approximate_max_probabilities(self, number_of_treatments, context):
        # Use the current context to prepare the design matrix
        # Convert context to DataFrame for design matrix preparation

        # Prepare a DataFrame with current context
        context_df = pd.DataFrame(
            {
                self.treatment_name: list(range(number_of_treatments)),
                **{
                    col: [context.get(col, 0)] * number_of_treatments
                    for col in self.covariate_names
                },
            }
        )

        # If categorical covariates are specified, add those
        for col, categories in self.categorical_covariates.items():
            # Use the current context value or default to 0
            context_df[col] = context.get(col, 0)

        # Prepare design matrix with current context
        x = self._prepare_design_matrix(context_df, number_of_treatments)

        # If no history, fall back to uniform sampling
        if not hasattr(self, "history") or len(self.history.observations) == 0:
            return [1 / number_of_treatments] * number_of_treatments

        # Convert history to DataFrame
        self.df = self.history.to_df()

        # If no historical data, initialize model with priors
        self.model = NIGLinearRegression(
            mu=self.mean, v=self.v, a=self.alpha, b=self.beta
        )

        # If we have historical data, fit the model
        if len(self.df) > 0:
            # Prepare historical design matrix
            x_history = self._prepare_design_matrix(self.df, number_of_treatments)

            # Fit the model with historical data
            self.model.fit(x_history, self.df[self.outcome_name])

        sample_size = 1000
        identity_matrix = numpy.eye(number_of_treatments)

        # Sample predictions using the current context
        samples = numpy.array(
            [
                self.model.predict(
                    self._create_prediction_points(
                        identity_matrix[intervention],
                        x[intervention, number_of_treatments:],  # Use context values
                    )
                ).rvs(size=sample_size)
                for intervention in range(number_of_treatments)
            ]
        )

        max_indices = numpy.argmax(samples, axis=0)
        bin_counts = numpy.bincount(max_indices, minlength=number_of_treatments)
        return bin_counts / numpy.sum(bin_counts)

    def _create_prediction_points(self, treatment_point, covariate_values=None):
        """
        Create prediction points by combining treatment dummies with covariate values.

        Parameters
        ----------
        treatment_point : numpy.ndarray
            One-hot encoded treatment vector
        covariate_values : numpy.ndarray, optional
            Mean or specific covariate values

        Returns
        -------
        numpy.ndarray
            Combined treatment and covariate values for prediction
        """
        if covariate_values is None:
            return treatment_point.reshape(1, -1)

        return numpy.hstack([treatment_point, covariate_values]).reshape(1, -1)

    def __str__(self):
        return f"BayesianLinearRegressionModel{self.name}"

    @property
    def debug_data(self):
        if not self.model:
            return {
                "mean": numpy.round(self.mean, 2),
                "v": numpy.round(self.v, 2),
                "alpha": numpy.round(self.alpha, 2),
                "beta": numpy.round(self.beta, 2),
            }
        return {
            "mean": numpy.round(self.model.mu, 2),
            "v": numpy.round(self.model.v, 2),
            "alpha": numpy.round(self.model.a, 2),
            "beta": numpy.round(self.model.b, 2),
        }
