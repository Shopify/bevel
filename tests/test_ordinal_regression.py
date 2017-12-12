import numpy as np
import os
import pandas as pd
import pytest

from eunomia.ordinal_regression import logistic
from eunomia.ordinal_regression import OrdinalRegression
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

from scipy import stats


eunomia_dir = os.path.dirname(os.path.realpath(__file__))
eunomia_root_dir = os.path.dirname(eunomia_dir)
filepath = os.path.join(eunomia_root_dir, 'data/ucla.dta')

ucla_data = pd.read_stata(filepath)
# Ordinal Logistic Regression. UCLA: Statistical Consulting Group.
# from https://stats.idre.ucla.edu/r/dae/ordinal-logistic-regression/
# (accessed 8 December, 2017).


@pytest.fixture
def X_ucla():
    return ucla_data[['pared', 'public', 'gpa']].values

@pytest.fixture
def y_ucla():
    return ucla_data['apply'].map({'unlikely': 1,'somewhat likely': 2,'very likely': 3}).values
   

def test_logistic():
    z = np.array([0., np.inf, -np.inf])
    expected = np.array([0.5, 1., 0.])
    assert_array_equal(logistic(z), expected)


class TestOrdinalRegression():

    def test_ucla_coef_(self, X_ucla, y_ucla):
        ordinal_regression = OrdinalRegression()
        ordinal_regression.fit(X_ucla, y_ucla)
        
        expected_coef_ = np.array([1.04769, -0.05879, 0.61594, 2.20391, 4.29936])
        assert_allclose(ordinal_regression.coef_, expected_coef_, rtol=0.01)

    def test_ucla_se_(self, X_ucla, y_ucla):              
        ordinal_regression = OrdinalRegression()
        ordinal_regression.fit(X_ucla, y_ucla)

        expected_se_ = np.array([0.2658, 0.2979, 0.2606, 0.7795, 0.8043])
        assert_allclose(ordinal_regression.se_, expected_se_, rtol=0.01)

    def test_ucla_z_values_(self, X_ucla, y_ucla):              
        ordinal_regression = OrdinalRegression()
        ordinal_regression.fit(X_ucla, y_ucla)

        expected_z_values_ = np.array([3.9418, -0.1974, 2.3632, 2.8272, 5.3453])
        assert_allclose(ordinal_regression._z_values(), expected_z_values_, rtol=0.01)

    def test_ucla_p_values_(self, X_ucla, y_ucla):              
        ordinal_regression = OrdinalRegression()
        ordinal_regression.fit(X_ucla, y_ucla)

        expected_p_values_ = np.array([8.087e-05, 8.435e-01, 1.812e-02, 4.696e-03, 9.027e-08])
        assert_allclose(ordinal_regression.p_values_, expected_p_values_, rtol=0.01)

    def test_jacobian(self):
        ordinal_regression = OrdinalRegression()
        ordinal_regression.n_attributes = 2
        ordinal_regression.n_classes = 5
        expected = np.array([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 1., 0., 0.],
            [0., 0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 1., 1.]
        ])
        assert_array_equal(expected, ordinal_regression._jacobian())

    def test_log_likelihood(self):
        ordinal_regression = OrdinalRegression()
        X = np.array([[1.0], [1.0], [1.0]])
        y = np.array([1,1,2])
        coefficients = np.array([1.0, 1.0])
        ordinal_regression.n_attributes = 1
        actual = ordinal_regression._log_likelihood(coefficients, X, y)
        expected = - 3.0 * np.log(0.5)
        assert actual == expected

    def test_clean_y(self):
        ordinal_regression = OrdinalRegression()
        y = np.array([1,3,5])
        expected = np.array([1,2,3])
        _, actual = ordinal_regression._clean(np.empty((1,1)), y)
        assert_array_equal(actual, expected)
        assert ordinal_regression.N == 1
        assert ordinal_regression.n_attributes == 1
        assert ordinal_regression.n_classes == 3