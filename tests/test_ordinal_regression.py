import numpy as np
import os
import pandas as pd
import pytest

from bevel.ordinal_regression import logistic
from bevel.ordinal_regression import OrdinalRegression
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

from scipy import stats


bevel_dir = os.path.dirname(os.path.realpath(__file__))
bevel_root_dir = os.path.dirname(bevel_dir)
filepath = os.path.join(bevel_root_dir, 'data/ucla.dta')

ucla_data = pd.read_stata(filepath)
# Ordinal Logistic Regression. UCLA: Statistical Consulting Group.
# from https://stats.idre.ucla.edu/r/dae/ordinal-logistic-regression/
# (accessed 8 December, 2017).


@pytest.fixture
def X_ucla():
    return ucla_data.drop("apply", axis=1)

@pytest.fixture
def y_ucla():
    return ucla_data['apply'].map({'unlikely': 1,'somewhat likely': 2,'very likely': 3})


def test_logistic():
    z = np.array([0., np.inf, -np.inf])
    expected = np.array([0.5, 1., 0.])
    assert_array_equal(logistic(z), expected)


class TestOrdinalRegression():

    def test_ucla_coef(self, X_ucla, y_ucla):
        orf = OrdinalRegression()
        orf.fit(X_ucla, y_ucla)
        expected_coef_ = np.array([1.04769, -0.05879, 0.61594, 2.20391, 4.29936])
        assert_allclose(orf.coef_, expected_coef_, rtol=0.01)

    def test_ucla_se(self, X_ucla, y_ucla):
        orf = OrdinalRegression()
        orf.fit(X_ucla, y_ucla)

        expected_se_ = np.array([0.2658, 0.2979, 0.2606, 0.7795, 0.8043])
        assert_allclose(orf.se_, expected_se_, rtol=0.01)

    def test_ucla_z_values(self, X_ucla, y_ucla):
        orf = OrdinalRegression()
        orf.fit(X_ucla, y_ucla)

        expected_z_values_ = np.array([3.9418, -0.1974, 2.3632, 2.8272, 5.3453])
        assert_allclose(orf._compute_z_values(), expected_z_values_, rtol=0.01)

    def test_ucla_p_values(self, X_ucla, y_ucla):
        orf = OrdinalRegression()
        orf.fit(X_ucla, y_ucla)

        expected_p_values_ = np.array([8.087e-05, 8.435e-01, 1.812e-02, 4.696e-03, 9.027e-08])
        assert_allclose(orf.p_values_, expected_p_values_, rtol=0.01)

    def test_compute_basis_change(self):
        orf = OrdinalRegression()
        orf.n_attributes = 2
        orf.n_classes = 5
        expected = np.array([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 1., 0., 0.],
            [0., 0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 1., 1.]
        ])
        assert_array_equal(expected, orf._compute_basis_change())

    def test_log_likelihood(self):
        orf = OrdinalRegression()
        X = np.array([[1.0], [1.0], [1.0]])
        y = np.array([1,1,2])
        coefficients = np.array([1.0, 1.0])
        orf.n_attributes = 1
        actual = orf._log_likelihood(coefficients, X, y)
        expected = - 3.0 * np.log(0.5)
        assert actual == expected

    def test_gradient(self):
        orf = OrdinalRegression()
        X = np.array([[1.0], [1.0], [1.0]])
        y = np.array([1,1,2])
        coefficients = np.array([1.0, 1.0])
        orf.n_attributes = 1
        orf.n_classes = 2
        actual = orf._gradient(coefficients, X, y)
        expected = np.array([0.5, -0.5])
        assert_array_equal(actual, expected)

    def test_prepare(self):
        orf = OrdinalRegression()
        y = np.array([1,3,5])
        _, actual = orf._prepare(np.empty((1,1)), y)
        expected = np.array([1,2,3])
        assert_array_equal(actual, expected)
        assert orf.attribute_names == ['column_0']
        assert orf.N == 1
        assert orf.n_attributes == 1
        assert orf.n_classes == 3

    def test_summary(self, X_ucla, y_ucla):
        orf = OrdinalRegression(significance=0.95)
        orf.fit(X_ucla, y_ucla)
        assert 'coef' in orf.summary
        assert 'se(coef)' in orf.summary
        assert 'p' in orf.summary
        assert 'lower 0.95' in orf.summary
        assert 'upper 0.95' in orf.summary

    def test_predict_returns_correct_mappings(self, X_ucla, y_ucla):
        y_ucla[y_ucla == 1] = -1
        orf = OrdinalRegression()
        orf.fit(X_ucla, y_ucla)
        assert orf.predict_class(X_ucla).min() == -1

    def test_predict_class_returns_correct_number_of_output_predictions(self, X_ucla, y_ucla):
        orf = OrdinalRegression()
        orf.fit(X_ucla, y_ucla)
        n = 10
        X_pred = np.random.randn(n, X_ucla.shape[1])
        assert orf.predict_class(X_pred).shape[0] == n

    def test_predict_probs_returns_correct_number_of_output_predictions(self, X_ucla, y_ucla):
        orf = OrdinalRegression()
        orf.fit(X_ucla, y_ucla)
        n = 10
        X_pred = np.random.randn(n, X_ucla.shape[1])
        assert orf.predict_probabilities(X_pred).shape == (n, 3)

    def test_predict_can_accept_single_row(self, X_ucla, y_ucla):
        orf = OrdinalRegression()
        orf.fit(X_ucla, y_ucla)
        X_pred = np.random.randn(X_ucla.shape[1])
        assert orf.predict_class(X_pred).shape[0] == 1
