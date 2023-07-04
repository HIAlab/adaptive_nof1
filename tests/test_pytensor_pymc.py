from adaptive_nof1.inference.bayes import GaussianAverageTreatmentEffect
import pytest

import pymc
import numpy


@pytest.fixture
def observations():
    return pymc.floatX([1, 2, 3, 4, 5])


@pytest.fixture
def two_dimensional_matrix():
    return pymc.floatX([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])


@pytest.fixture
def three_column_matrix():
    three_column_matrix = pymc.floatX([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    assert three_column_matrix.shape == (5, 3)
    return three_column_matrix

@pytest.fixture
def three_dimensional_matrix():
    three_dimensional_matrix = pymc.floatX([[[1, 2], [3, 4], [5,6]], [[7, 8], [9, 10], [11, 12]]])
    assert three_dimensional_matrix.shape == (2, 3, 2)
    return three_dimensional_matrix


@pytest.fixture
def indices():
    return pymc.intX([0, 1, 0, 2, 0])


def test_create_row_from_column_indexes(three_column_matrix, indices):
    result = indices.choose(three_column_matrix.swapaxes(0,1))
    assert result.shape == (5,)
    assert list(result) == (
            [1, 5, 7, 12, 13]
    )

def test_model():
    n_observations = 3
    n_coefficients = 2
    n_treatments = 2

    # Let's assume two different treatment options with two different coefficients
    slopes = pymc.floatX([[0, 1], [2, 3]])
    assert slopes.shape == (n_treatments, n_coefficients)

    # We have 3 observations:
    observations = pymc.floatX([0, 1, 2])

    # And the applied treatments are
    treatment_indices = pymc.intX([1, 0, 1])

    # And the values of the coefficients are:
    coefficients = pymc.floatX([[0, 1, 2], [2, 3, 4], [4,5, 6]])
    # However, for this particular treatment, we only want to look at the coefficients 2 and 3:
    coefficient_indices = pymc.intX([1, 2])
    coefficients_for_treatment = coefficients[:, coefficient_indices]

    assert coefficients_for_treatment.shape == (n_observations, n_coefficients)
    assert numpy.array_equal(coefficients_for_treatment, [[1,2], [3, 4], [5,6]])

    # We now want to multiply the coefficients with the correct slopes per treatment
    slopes_for_treatments = slopes[treatment_indices]

    assert slopes_for_treatments.shape == (n_observations, n_coefficients)
    assert numpy.array_equal(slopes_for_treatments, [[2,3], [0,1], [2,3]])

    print(pymc.math.dot(coefficients_for_treatment, slopes_for_treatments.T).eval())
    coefficient_summand = pymc.math.dot(coefficients_for_treatment, slopes_for_treatments.T).diagonal()

    print(coefficient_summand.eval())
    assert coefficient_summand.shape.eval() == n_observations
    assert numpy.array_equal(coefficient_summand.eval(), [8, 4, 28])

def test_create_row_from_column_indexes_three_dimensional(three_dimensional_matrix):
    # This is comparable to a case where the most inner sequence is a number of coefficients,
    # the indices are the treatments,
    # and we want to select the correct coefficients for each treatment given
    indices = pymc.intX([2, 1])
    result = three_dimensional_matrix[:, indices].diagonal(axis1=0, axis2=1).T
    print("result\n",result)
    assert result.shape == (2,2)
    assert list(result) == (
            [[5,6], [9,10]]
    )
