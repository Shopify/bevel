import numpy as np
import os
import pandas as pd
import pytest

from bevel.linear_ordinal_regression import LinearOrdinalRegression
from bevel.linear_ordinal_regression import OrderedLogit
from bevel.linear_ordinal_regression import OrderedProbit

from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from scipy.special import expit

from pandas.testing import assert_frame_equal

bevel_dir = os.path.dirname(os.path.realpath(__file__))
bevel_root_dir = os.path.dirname(bevel_dir)
ucla_filepath = os.path.join(bevel_root_dir, 'data/ucla.dta')
academy_filepath = os.path.join(bevel_root_dir, 'data/econometric_academy.dta')

ucla_data = pd.read_stata(ucla_filepath)
# Ordinal Logistic Regression. UCLA: Statistical Consulting Group.
# from https://stats.idre.ucla.edu/r/dae/ordinal-logistic-regression/
# (accessed 8 December, 2017).

academy_data = pd.read_stata(academy_filepath)
# Katchova, Ani L. (2013) Econometrics Academy [Website and YouTube Channel]
# Retrieved from Econometrics Academy Website: https://sites.google.com/site/econometricsacademy/
# (accessed 10 March, 2018).

@pytest.fixture
def X_ucla():
    return ucla_data.drop('apply', axis=1)

@pytest.fixture
def y_ucla():
    return ucla_data['apply'].map({'unlikely': 1,'somewhat likely': 2,'very likely': 3})

@pytest.fixture
def X_academy():
    return academy_data.drop(['healthstatus1', 'healthstatus'], axis=1)

@pytest.fixture
def y_academy():
    return academy_data['healthstatus1']

@pytest.fixture
def sample_lor():
    lor = OrderedLogit(significance=0.9)
    lor.alpha_ = np.array([1, 1])
    lor.beta_ = np.array([1, 1, 1])
    lor._y_dict = {1:1, 2:2, 3:7}
    lor.n_attributes = 3
    lor.n_classes = 3
    lor.se_ = np.array([1, 1, 1, 1, 1])
    lor.p_values_ = np.array([0, 0, 0, 0, 0])
    lor.attribute_names = pd.DataFrame(
        ['attribute_1', 'attribute_2', 'attribute_3'], 
        columns=['attribute names']
    )
    return lor

class TestLinearOrdinalRegression():

    def test_compute_basis_change(self):
        lor = LinearOrdinalRegression(None, None)
        lor.n_attributes = 2
        lor.n_classes = 5
        
        expected_P = np.array([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 1., 0., 0.],
            [0., 0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 1., 1.]
        ])
        assert_array_equal(expected_P, lor._compute_basis_change())

    def test_prepare_y(self):
        lor = LinearOrdinalRegression(None, None)
        y = np.array([1,3,5])
        y_data = lor._prepare_y(y)
        
        assert_array_equal(y_data, np.array([1,2,3]))
        assert lor.n_classes == 3

    def test_prepare_X(self):
        lor = LinearOrdinalRegression(None, None)
        X = np.array([[-1, 0, 1], [0, 1, -1], [1, -1, 0], [-3, 3, -3], [3, -3, 3]])
        X_data, X_scale, X_mean, X_std = lor._prepare_X(X)
        
        assert_array_equal(X_data, X)
        assert_array_equal(X_scale, X / 2.0)
        assert_array_equal(X_mean, np.array([0, 0, 0]))
        assert_array_equal(X_std, np.array([2, 2, 2]))
        assert lor.N == 5
        assert lor.n_attributes == 3

    def test_vanishing_variance_raises_error(self):
        lor = LinearOrdinalRegression(None, None)
        X = np.array([[1,1], [1,2], [1,3]])
        with pytest.raises(ValueError):
            lor._prepare_X(X)

    def test_get_column_names_df(self):
        X = pd.DataFrame(columns=['a', 'b'])
        
        expected = pd.DataFrame(['a', 'b'], columns=['attribute names'])
        assert_frame_equal(LinearOrdinalRegression(None, None)._get_column_names(X), expected)

    def test_get_column_names_array(self):
        X = np.array(None)
        lor = LinearOrdinalRegression(None, None)
        lor.n_attributes = 2
        
        expected = pd.DataFrame(['column_1', 'column_2'], columns=['attribute names'])
        assert_frame_equal(lor._get_column_names(X), expected)

    def test_log_likelihood(self):
        lor = LinearOrdinalRegression(None, None)
        X = np.array([[1.0], [1.0], [1.0]])
        y = np.array([1, 1, 2])
        coefficients = np.array([1.0, 1.0])
        lor.n_attributes = 1
        lor.link = expit
        assert lor._log_likelihood(coefficients, X, y) == - 3.0 * np.log(0.5)

    def test_summary(self, X_ucla, y_ucla, sample_lor):
        lor = sample_lor
        assert 'beta' in lor.summary
        assert 'se(beta)' in lor.summary
        assert 'p' in lor.summary
        assert 'lower 0.90' in lor.summary
        assert 'upper 0.90' in lor.summary

    def test_predict_linear_product(self, X_ucla, y_ucla):
        lor = LinearOrdinalRegression(None, None)
        lor.beta_ = np.array([1, -1, 2])
        assert lor.predict_linear_product(np.ones(3)) == 1 + -1 + 2
        assert lor.predict_linear_product(np.array([1, 0, 1.5])) == 1*1 + -1*0 + 2*1.5

    def test_predict_probabilities_output_size(self, X_ucla, y_ucla, sample_lor):
        n = 10
        X_pred = np.random.randn(n, X_ucla.shape[1])
        assert sample_lor.predict_probabilities(X_pred).shape == (n, 3)

    def test_predict_class_returns_correct_mappings(self, X_ucla, y_ucla, sample_lor):
        X = np.array([[0, 0, 0], [1, 1, 1]])
        assert_array_equal(sample_lor.predict_class(X), np.array([1, 7]))

    def test_predict_class_output_size(self, X_ucla, y_ucla, sample_lor):
        n = 10
        X_pred = np.random.randn(n, X_ucla.shape[1])
        assert sample_lor.predict_class(X_pred).shape[0] == n

    def test_predict_can_accept_single_row(self, X_ucla, y_ucla, sample_lor):
        X_pred = np.random.randn(X_ucla.shape[1])
        assert sample_lor.predict_class(X_pred).shape[0] == 1
        assert sample_lor.predict_linear_product(X_pred).shape[0] == 1
        assert sample_lor.predict_probabilities(X_pred).shape[0] == 1

    def test_score(self, X_ucla, y_ucla):
        X = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
        ])
        y = [1, 2, 3]
        lor = LinearOrdinalRegression(None, None)
        lor.beta_ = np.array([1.0, 0.5, -1.0])
        assert lor._compute_score(X, y) == (0.0 - 3.0) / (3.0 * 2.0 / 2.0)


class TestOrderedLogit():

    def test_ucla_coef(self, X_ucla, y_ucla):
        ol = OrderedLogit()
        ol.fit(X_ucla, y_ucla)
        
        expected_coef_ = np.array([1.04769, -0.05879, 0.61594, 2.20391, 4.29936])
        assert_allclose(ol.coef_, expected_coef_, rtol=0.01)

    def test_ucla_se(self, X_ucla, y_ucla):
        ol = OrderedLogit()
        ol.fit(X_ucla, y_ucla)
        
        expected_se_ = np.array([0.2658, 0.2979, 0.2606, 0.7795, 0.8043])
        assert_allclose(ol.se_, expected_se_, rtol=0.01)

    def test_ucla_z_values(self, X_ucla, y_ucla):
        ol = OrderedLogit()
        ol.fit(X_ucla, y_ucla)
        
        expected_z_values_ = np.array([3.9418, -0.1974, 2.3632, 2.8272, 5.3453])
        assert_allclose(ol._compute_z_values(), expected_z_values_, rtol=0.01)

    def test_ucla_p_values(self, X_ucla, y_ucla):
        ol = OrderedLogit()
        ol.fit(X_ucla, y_ucla)

        expected_p_values_ = np.array([8.087e-05, 8.435e-01, 1.812e-02, 4.696e-03, 9.027e-08])
        assert_allclose(ol.p_values_, expected_p_values_, rtol=0.01)

    def test_gradient(self):
        ol = OrderedLogit()
        X = np.array([[1.0], [1.0], [1.0]])
        y = np.array([1,1,2])
        coefficients = np.array([1.0, 1.0])
        ol.n_attributes = 1
        ol.n_classes = 2
        
        expected = np.array([0.5, -0.5])
        assert_array_equal(ol._gradient(coefficients, X, y), expected)

class TestOrderedProbit():

    def test_academy_coef(self, X_academy, y_academy):
        op = OrderedProbit()
        op.fit(X_academy, y_academy)

        expected_coef_ = np.array([-0.0171681, 0.1654079, -0.0315288, -0.7945455, 0.5459371])
        assert_allclose(op.coef_, expected_coef_, rtol=0.01)

    def test_academy_se(self, X_academy, y_academy):
        op = OrderedProbit()
        op.fit(X_academy, y_academy)
        
        expected_se_ = np.array([0.000983, 0.0128, 0.00238, 0.1146, 0.1149])
        assert_allclose(op.se_, expected_se_, rtol=0.01)

