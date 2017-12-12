import numpy as np

from numdifftools import Hessian
from numpy.linalg import inv
from scipy import optimize
from scipy.linalg import block_diag

from scipy import stats


def logistic(z):
    positive_z = z > 0
    logistic = np.zeros_like(z, dtype=np.float)
    logistic[positive_z] = 1.0 / (1 + np.exp(-z[positive_z]))
    exp_z = np.exp(z[~positive_z])
    logistic[~positive_z] = exp_z / (1.0 + exp_z)
    return logistic


class OrdinalRegression():

    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = self._clean(X, y)

        beta_guess = np.zeros(self.n_attributes)
        gamma_guess = np.ones(self.n_classes - 1)
        bounds = [(None, None)] * (self.n_attributes + 1) + [(0, None)] * (self.n_classes - 2)

        optimization = optimize.minimize(
            self._log_likelihood, 
            np.append(beta_guess, gamma_guess),
            args=(X, y),
            bounds=bounds
        )

        self._set_coefficients(optimization.x)
        self.se_ = self._standard_errors(X, y)
        self.p_values_ = self._p_values()
        return self

    def _clean(self, X, y):
        X = np.asarray(X)
        self.N, self.n_attributes = X.shape

        y = np.asarray(y).astype(np.int)
        y_values = np.sort(np.unique(y))
        self.n_classes = len(y_values)
        y_range = np.arange(1, self.n_classes + 1)
        y = np.vectorize(dict(zip(y_values, y_range)).get)(y)

        return X, y

    def _log_likelihood(self, coefficients, X, y):
        beta = coefficients[:self.n_attributes]
        gamma = coefficients[self.n_attributes:]

        _alpha = np.cumsum(gamma)
        _alpha = np.insert(_alpha, 0, -np.inf)
        _alpha = np.append(_alpha, np.inf)

        z_plus = _alpha[y] - X.dot(beta)
        z_minus = _alpha[y-1] - X.dot(beta)
        return - 1.0 * np.sum(np.log(logistic(z_plus) - logistic(z_minus)))

    def _set_coefficients(self, optimization_x):
        self.gamma_ = optimization_x[self.n_attributes:]
        self.beta_ = optimization_x[:self.n_attributes]
        self.alpha_ = np.cumsum(self.gamma_)
        self.coef_ = np.append(self.beta_, self.alpha_)

    def _standard_errors(self, X, y):
        hessian_function = Hessian(self._log_likelihood, method='forward')
        H = hessian_function(np.append(self.beta_, self.gamma_), X, y)
        J = self._jacobian()
        return np.sqrt(np.diagonal(J.dot(inv(H)).dot(J.T)))

    def _jacobian(self):
        upper_left_diagonal = np.identity(self.n_attributes)
        lower_right_triangular = np.tril(np.ones((self.n_classes - 1, self.n_classes-1)))
        return block_diag(upper_left_diagonal, lower_right_triangular)

    def _z_values(self):
        return self.coef_ / self.se_

    def _p_values(self):
        z_magnitudes = np.abs(self._z_values())
        p_values = 1 - stats.norm.cdf(z_magnitudes) + stats.norm.cdf(-z_magnitudes)
        return p_values