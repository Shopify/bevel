import numpy as np

from scipy import optimize
from scipy.linalg import block_diag
from numpy.linalg import inv
from numdifftools import Hessian

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

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
        X = np.asarray(X)
        y = np.asarray(y).astype(np.int)

        self.n_classes = np.max(y)
        self.N, self.n_attributes = X.shape

        y_values = np.sort(np.unique(y))
        y_range = np.arange(1, self.n_classes + 1)
        
        #TODO: clean the data instead of raising an error
        if np.any(y_values != y_range):
            raise ValueError('Values in y must be in {}, but received {}'.format(y_range, y_values))

        beta_guess = np.zeros(self.n_attributes)
        gamma_guess = np.ones(self.n_classes - 1)
        bounds = [(None, None)] * (self.n_attributes + 1) + [(0, None)] * (self.n_classes - 2)

        optimization = optimize.minimize(
            self.log_likelihood, 
            np.append(beta_guess, gamma_guess),
            args=(X, y),
            bounds=bounds
        )

        self.gamma_ = optimization.x[self.n_attributes:]
        self.beta_ = optimization.x[:self.n_attributes]
        self.alpha_ = np.cumsum(self.gamma_)
        self.coef_ = np.append(self.beta_, self.alpha_)
        self.se_ = self.standard_errors(X, y)
        return self

    def standard_errors(self, X, y):
        hessian_function = Hessian(self.log_likelihood, method='forward')
        H = hessian_function(np.append(self.beta_, self.gamma_), X, y)
        J = self.jacobian()
        return np.sqrt(np.diagonal(J.dot(inv(H)).dot(J.T)))

    def jacobian(self):
        upper_left_diagonal = np.identity(self.n_attributes)
        lower_right_triangular = np.tril(np.ones((self.n_classes - 1, self.n_classes-1), dtype=np.float))
        return block_diag(upper_left_diagonal, lower_right_triangular)

    def log_likelihood(self, coefficients, X, y):
        beta = coefficients[:self.n_attributes]
        gamma = coefficients[self.n_attributes:]

        _alpha = np.cumsum(gamma)
        _alpha = np.insert(_alpha, 0, -np.inf)
        _alpha = np.append(_alpha, np.inf)

        z_plus = _alpha[y] - X.dot(beta)
        z_minus = _alpha[y-1] - X.dot(beta)
        return - 1.0 * np.sum(np.log(logistic(z_plus) - logistic(z_minus)))
