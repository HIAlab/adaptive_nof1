from adaptive_nof1.inference.bayesian_linear_regression import (
    BayesianLinearRegressionModel,
    NIGLinearRegression,
)
import pytest
import numpy

from adaptive_nof1.helpers import array_almost_equal
from .helper_functions import outcomes_to_history


class TestNIGLinearRegression:
    """Tests for the NIGLinearRegression class"""

    def test_init(self):
        """Test initialization of NIGLinearRegression"""
        mu = numpy.array([0, 0])
        v = numpy.eye(2)
        a = 1
        b = 1

        model = NIGLinearRegression(mu=mu, v=v, a=a, b=b)

        assert numpy.array_equal(model.mu, mu)
        assert numpy.array_equal(model.v, v)
        assert model.a == a
        assert model.b == b

    def test_fit_updates_parameters(self):
        """Test that fit updates the posterior parameters"""
        mu = numpy.array([0, 0])
        v = numpy.eye(2)
        a = 1
        b = 1

        model = NIGLinearRegression(mu=mu, v=v, a=a, b=b)

        # Simple training data
        x_tr = numpy.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        y_tr = numpy.array([1, 2, 1.5, 2.5])

        model.fit(x_tr, y_tr)

        # Parameters should have changed after fitting
        assert not numpy.array_equal(model.mu, mu)
        assert not numpy.array_equal(model.v, v)
        assert model.a != a
        assert model.b != b

    def test_predict_returns_distribution(self):
        """Test that predict returns a scipy distribution"""
        mu = numpy.array([1, 2])
        v = numpy.eye(2) * 0.5
        a = 10
        b = 10

        model = NIGLinearRegression(mu=mu, v=v, a=a, b=b)

        x_te = numpy.array([[1, 0], [0, 1]])
        dist = model.predict(x_te)

        # Should be able to sample from the distribution
        samples = dist.rvs(size=100)
        assert len(samples) == 100

    def test_get_conditional_beta(self):
        """Test get_conditional_beta returns multivariate normal"""
        mu = numpy.array([1, 2])
        v = numpy.eye(2)
        a = 1
        b = 1

        model = NIGLinearRegression(mu=mu, v=v, a=a, b=b)
        dist = model.get_conditional_beta(sigma2=1.0)

        # Should be able to sample from the distribution
        samples = dist.rvs(size=10)
        assert samples.shape == (10, 2)

    def test_get_marginal_sigma2(self):
        """Test get_marginal_sigma2 returns inverse gamma distribution"""
        mu = numpy.array([1, 2])
        v = numpy.eye(2)
        a = 2
        b = 3

        model = NIGLinearRegression(mu=mu, v=v, a=a, b=b)
        dist = model.get_marginal_sigma2()

        # Should be able to sample from the distribution
        samples = dist.rvs(size=100)
        assert len(samples) == 100
        # All samples should be positive (variance)
        assert numpy.all(samples > 0)


class TestBayesianLinearRegressionModel:
    """Tests for the BayesianLinearRegressionModel class"""

    def test_init_default(self):
        """Test default initialization"""
        model = BayesianLinearRegressionModel()

        assert model.treatment_name == "treatment"
        assert model.outcome_name == "outcome"
        assert model.mean == 0
        assert model.v == 0
        assert model.alpha == 0
        assert model.beta == 0

    def test_init_with_parameters(self):
        """Test initialization with custom parameters"""
        mean = numpy.array([2.5, 2.5])
        v = numpy.eye(2) * 0.5
        alpha = 10
        beta = 10

        model = BayesianLinearRegressionModel(mean=mean, v=v, alpha=alpha, beta=beta)

        assert numpy.array_equal(model.mean, mean)
        assert numpy.array_equal(model.v, v)
        assert model.alpha == alpha
        assert model.beta == beta

    def test_str(self):
        """Test string representation"""
        model = BayesianLinearRegressionModel()
        assert str(model) == "BayesianLinearRegression"

    def test_update_posterior(self):
        """Test that update_posterior stores history"""
        mean = numpy.array([0, 0])
        v = numpy.eye(2)
        alpha = 1
        beta = 1

        model = BayesianLinearRegressionModel(mean=mean, v=v, alpha=alpha, beta=beta)

        treatments = [0, 1, 0, 1]
        outcomes = [1, 2, 1.5, 2.5]
        history = outcomes_to_history(treatments, outcomes)

        model.update_posterior(history, number_of_treatments=2)

        assert model.history == history

    def test_get_upper_confidence_bounds_not_implemented(self):
        """Test that get_upper_confidence_bounds raises AssertionError"""
        model = BayesianLinearRegressionModel()

        with pytest.raises(AssertionError, match="Not implemented"):
            model.get_upper_confidence_bounds("test_variable")

    def test_debug_data_before_fit(self):
        """Test debug_data returns initial parameters before fitting"""
        mean = numpy.array([2.5, 2.5])
        v = numpy.eye(2) * 0.5
        alpha = 10
        beta = 10

        model = BayesianLinearRegressionModel(mean=mean, v=v, alpha=alpha, beta=beta)

        data = model.debug_data()

        assert numpy.array_equal(data["mean"], mean)
        assert numpy.array_equal(data["v"], v)
        assert data["alpha"] == alpha
        assert data["beta"] == beta

    def test_debug_data_after_fit(self):
        """Test debug_data returns updated parameters after fitting"""
        mean = numpy.array([0, 0])
        v = numpy.eye(2)
        alpha = 1
        beta = 1

        model = BayesianLinearRegressionModel(mean=mean, v=v, alpha=alpha, beta=beta)

        treatments = [0, 1, 0, 1]
        outcomes = [1, 2, 1.5, 2.5]
        history = outcomes_to_history(treatments, outcomes)

        model.update_posterior(history, number_of_treatments=2)
        model.approximate_max_probabilities(2, {"t": len(outcomes)})

        data = model.debug_data()

        # Parameters should be from the fitted model
        assert "mean" in data
        assert "v" in data
        assert "alpha" in data
        assert "beta" in data

    @pytest.mark.sampling
    def test_approximate_max_probabilities_empty_history(self):
        """Test approximate_max_probabilities with empty history"""
        mean = numpy.array([0, 0])
        v = numpy.eye(2)
        alpha = 1
        beta = 1

        model = BayesianLinearRegressionModel(mean=mean, v=v, alpha=alpha, beta=beta)

        history = outcomes_to_history([], [])
        model.update_posterior(history, number_of_treatments=2)
        p = model.approximate_max_probabilities(2, {"t": 0})

        assert array_almost_equal(p, [0.5, 0.5], epsilon=0.02)

    @pytest.mark.sampling
    @pytest.mark.parametrize(
        "treatments, priors, outcomes, expected",
        [
            # Equal outcomes for both treatments -> 50/50 split
            (
                [0, 0, 1, 1],
                (numpy.array([2.5, 2.5]), numpy.eye(2) * 0.5, 10, 10),
                [3, 3, 3, 3],
                [0.5, 0.5],
            ),
            # Treatment 1 has better outcomes -> higher probability
            (
                [0, 0, 1, 1],
                (numpy.array([2.5, 2.5]), numpy.eye(2) * 0.5, 10, 10),
                [3, 3, 5, 4.5],
                [0.25, 0.75],
            ),
            # Treatment 0 has better outcomes -> higher probability
            (
                [0, 0, 1, 1],
                (numpy.array([2.5, 2.5]), numpy.eye(2) * 0.5, 10, 10),
                [5, 4.5, 3, 3],
                [0.75, 0.25],
            ),
        ],
    )
    def test_approximate_max_probabilities(
        self, treatments, priors, outcomes, expected
    ):
        """Test approximate_max_probabilities with various scenarios"""
        number_of_treatments = 2
        mean, v, alpha, beta = priors

        model = BayesianLinearRegressionModel(mean=mean, v=v, alpha=alpha, beta=beta)

        model.update_posterior(
            outcomes_to_history(treatments, outcomes), number_of_treatments
        )
        p = model.approximate_max_probabilities(
            number_of_treatments, {"t": len(outcomes)}
        )

        assert array_almost_equal(p, expected, epsilon=0.02)

    def test_approximate_max_probabilities_wrong_mean_shape(self):
        """Test that wrong mean shape raises assertion"""
        mean = numpy.array([0, 0])
        v = numpy.eye(2)
        alpha = 1
        beta = 1

        model = BayesianLinearRegressionModel(mean=mean, v=v, alpha=alpha, beta=beta)

        treatments = [0, 1, 2]
        outcomes = [1, 2, 3]
        history = outcomes_to_history(treatments, outcomes)

        model.update_posterior(history, number_of_treatments=3)

        with pytest.raises(AssertionError, match="Wrong shape for mean"):
            model.approximate_max_probabilities(3, {"t": len(outcomes)})

    def test_approximate_max_probabilities_wrong_v_shape(self):
        """Test that wrong v shape raises assertion"""
        mean = numpy.array([0, 0, 0])
        v = numpy.eye(2)  # Wrong shape
        alpha = 1
        beta = 1

        model = BayesianLinearRegressionModel(mean=mean, v=v, alpha=alpha, beta=beta)

        treatments = [0, 1, 2]
        outcomes = [1, 2, 3]
        history = outcomes_to_history(treatments, outcomes)

        model.update_posterior(history, number_of_treatments=3)

        with pytest.raises(AssertionError, match="Wrong shape for v"):
            model.approximate_max_probabilities(3, {"t": len(outcomes)})

    @pytest.mark.sampling
    def test_posterior(self):
        """Test posterior returns a distribution for a specific intervention"""
        mean = numpy.array([1, 2])
        v = numpy.eye(2) * 0.5
        alpha = 10
        beta = 10

        model = BayesianLinearRegressionModel(mean=mean, v=v, alpha=alpha, beta=beta)

        treatments = [0, 1, 0, 1]
        outcomes = [1, 2, 1.5, 2.5]
        history = outcomes_to_history(treatments, outcomes)

        model.update_posterior(history, number_of_treatments=2)
        model.approximate_max_probabilities(2, {"t": len(outcomes)})

        # Get posterior for intervention 0
        dist = model.posterior(0)

        # Should be able to sample from it
        samples = dist.rvs(size=100)
        assert len(samples) == 100

    @pytest.mark.sampling
    def test_series(self):
        """Test series returns outcomes for a specific intervention"""
        mean = numpy.array([0, 0])
        v = numpy.eye(2)
        alpha = 1
        beta = 1

        model = BayesianLinearRegressionModel(mean=mean, v=v, alpha=alpha, beta=beta)

        treatments = [0, 1, 0, 1]
        outcomes = [1, 2, 1.5, 2.5]
        history = outcomes_to_history(treatments, outcomes)

        model.update_posterior(history, number_of_treatments=2)
        model.approximate_max_probabilities(2, {"t": len(outcomes)})

        # Get series for intervention 0
        series_0 = model.series(0)
        assert len(series_0) == 2
        assert list(series_0) == [1, 1.5]

        # Get series for intervention 1
        series_1 = model.series(1)
        assert len(series_1) == 2
        assert list(series_1) == [2, 2.5]

    @pytest.mark.sampling
    def test_with_more_data_points(self):
        """Test model with more data points"""
        mean = numpy.array([0, 0])
        v = numpy.eye(2) * 0.5
        alpha = 10
        beta = 10

        model = BayesianLinearRegressionModel(mean=mean, v=v, alpha=alpha, beta=beta)

        # Treatment 1 is consistently better
        treatments = [0, 1] * 10
        outcomes = [3] * 10 + [5] * 10

        model.update_posterior(
            outcomes_to_history(treatments, outcomes), number_of_treatments=2
        )
        p = model.approximate_max_probabilities(2, {"t": len(outcomes)})

        # With more data, should be more confident that treatment 1 is better
        assert p[1] > 0.8
        assert p[0] < 0.2

    @pytest.mark.sampling
    def test_three_treatments(self):
        """Test model with three treatments"""
        mean = numpy.array([0, 0, 0])
        v = numpy.eye(3) * 0.5
        alpha = 10
        beta = 10

        model = BayesianLinearRegressionModel(mean=mean, v=v, alpha=alpha, beta=beta)

        treatments = [0, 1, 2, 0, 1, 2]
        outcomes = [3, 4, 5, 3, 4, 5]

        model.update_posterior(
            outcomes_to_history(treatments, outcomes), number_of_treatments=3
        )
        p = model.approximate_max_probabilities(3, {"t": len(outcomes)})

        # Treatment 2 should be best
        assert p[2] > p[1] > p[0]
        assert array_almost_equal(p, [0.1, 0.3, 0.6], epsilon=0.15)

    def test_init_with_covariates(self):
        """Test initialization with covariate names"""
        mean = numpy.array([0, 0])
        v = numpy.eye(2)
        alpha = 1
        beta = 1
        covariates = ["age", "weight"]

        model = BayesianLinearRegressionModel(
            mean=mean, v=v, alpha=alpha, beta=beta, covariate_names=covariates
        )

        assert model.covariate_names == covariates

    def test_prepare_design_matrix_without_covariates(self):
        """Test design matrix preparation without covariates"""
        mean = numpy.array([0, 0])
        v = numpy.eye(2)
        alpha = 1
        beta = 1

        model = BayesianLinearRegressionModel(
            mean=mean, v=v, alpha=alpha, beta=beta
        )

        treatments = [0, 1, 0, 1]
        outcomes = [1, 2, 1.5, 2.5]
        history = outcomes_to_history(treatments, outcomes)

        model.update_posterior(history, number_of_treatments=2)
        df = history.to_df()
        
        design_matrix = model._prepare_design_matrix(df, number_of_treatments=2)
        
        # Should have 4 rows and 2 columns (treatment dummies)
        assert design_matrix.shape == (4, 2)
        # First row should be [1, 0] for treatment 0
        assert numpy.array_equal(design_matrix[0], [1, 0])
        # Second row should be [0, 1] for treatment 1
        assert numpy.array_equal(design_matrix[1], [0, 1])

    def test_create_prediction_points_without_covariates(self):
        """Test prediction point creation without covariates"""
        mean = numpy.array([0, 0])
        v = numpy.eye(2)
        alpha = 1
        beta = 1

        model = BayesianLinearRegressionModel(
            mean=mean, v=v, alpha=alpha, beta=beta
        )

        treatment_point = numpy.array([1, 0])
        pred_point = model._create_prediction_points(treatment_point)

        # Should return treatment point unchanged
        assert pred_point.shape == (1, 2)
        assert numpy.array_equal(pred_point[0], [1, 0])

    def test_create_prediction_points_with_covariates(self):
        """Test prediction point creation with covariates"""
        mean = numpy.array([0, 0, 0])  # 2 treatments + 1 covariate
        v = numpy.eye(3)
        alpha = 1
        beta = 1

        model = BayesianLinearRegressionModel(
            mean=mean, v=v, alpha=alpha, beta=beta, covariate_names=["age"]
        )

        treatment_point = numpy.array([1, 0])
        covariate_values = numpy.array([50.0])
        pred_point = model._create_prediction_points(treatment_point, covariate_values)

        # Should combine treatment and covariate
        assert pred_point.shape == (1, 3)
        assert numpy.array_equal(pred_point[0], [1, 0, 50.0])

