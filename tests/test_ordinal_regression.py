import pytest
import numpy as np

import pandas as pd

from numpy.testing import assert_array_equal

from eunomia.ordinal_regression import OrdinalRegression
from eunomia.ordinal_regression import logistic


@pytest.fixture
def X():
    return np.array([
        [1.0, 1.0],
        [0.0, 2.0],
        [1.0, 5.0],
        [0.0, 10.0],
        [0.2, 11.0],
        [2.0, 4.0]
    ])

@pytest.fixture
def y():
    return np.array([1, 2, 3, 3, 1, 2])


def test_logistic():
    z = np.array([0.0, np.inf, -np.inf])
    expected = np.array([0.5, 1., 0.])
    assert_array_equal(logistic(z), expected)


class TestOrdinalRegression():

    def test_fit_returns_increasing_alphas(self, X, y):
        ordinal_regression = OrdinalRegression()
        ordinal_regression.fit(X, y)
        assert np.all(np.diff(ordinal_regression.alpha_) > 0)

    def test_zero_y_raises_value_error(self, X, y):
        pass

    def test_ucla_data(self):

        SCORE_MAP = {
            'unlikely': 1,
            'somewhat likely': 2,
            'very likely': 3
        }

        ucla_data = pd.read_stata('ucla.dta')
        y = ucla_data['apply'].map(SCORE_MAP).values
        X = ucla_data[['pared', 'public', 'gpa']].values
        
        ordinal_regression = OrdinalRegression()
        ordinal_regression.fit(X, y)
