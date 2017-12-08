import pytest
import numpy as np

import pandas as pd

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from eunomia.ordinal_regression import OrdinalRegression
from eunomia.ordinal_regression import logistic


ucla_data = pd.read_stata('ucla.dta')
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
        assert_array_almost_equal(ordinal_regression.coef_, expected_coef_, decimal=3)

    def test_ucla_se_(self, X_ucla, y_ucla):              
        ordinal_regression = OrdinalRegression()
        ordinal_regression.fit(X_ucla, y_ucla)

        expected_se_ = np.array([0.2658, 0.2979, 0.2606, 0.7795, 0.8043])
        assert_array_almost_equal(ordinal_regression.se_, expected_se_, decimal=3)

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
        assert_array_equal(expected, ordinal_regression.jacobian())

    def test_log_likelihood(self):
        ordinal_regression = OrdinalRegression()
        X = np.array([[1.0], [1.0], [1.0]])
        y = np.array([1,1,2])
        coefficients = np.array([1.0, 1.0])
        ordinal_regression.n_attributes = 1
        actual = ordinal_regression.log_likelihood(coefficients, X, y)
        expected = - 3.0 * np.log(0.5)
        assert actual == expected   
            