import numpy as np
from scipy import optimize
from sklearn import base


def logistic(z):
    positive_z = z > 0
    logistic = np.zeros_like(z, dtype=np.float)
    logistic[positive_z] = 1.0 / (1 + np.exp(-z[positive_z]))
    exp_z = np.exp(z[~positive_z])
    logistic[~positive_z] = exp_z / (1.0 + exp_z)
    return logistic


class OrdinalRegression(base.BaseEstimator):

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(np.int)

        self.n_classes = np.max(y)
        self.N, self.n_attributes = X.shape

        y_values = np.sort(np.unique(y))
        y_range = np.arange(1, self.n_classes + 1)
        if np.any(y_values != y_range):
            raise ValueError('Values in y must be in {}, but received {}'.format(y_range, y_values))

        beta_guess = np.zeros(self.n_attributes)
        gamma_guess = np.ones(self.n_classes - 1)
        x0 = np.append(beta_guess, gamma_guess)
        bounds = [(None, None)] * (self.n_attributes + 1) + [(0, None)] * (self.n_classes - 2)

        optimization = optimize.minimize(
            self.log_likelihood, 
            x0,
            args=(X, y),
            bounds=bounds
        )
        
        self.beta_ = optimization.x[:self.n_attributes]
        self.alpha_ = np.cumsum(optimization.x[self.n_attributes:])
        return self

    def log_likelihood(self, x0, X, y):
        beta = x0[:self.n_attributes]
        gamma = x0[self.n_attributes:]

        alphas = np.cumsum(gamma)
        alphas = np.insert(alphas, 0, -np.inf)
        alphas = np.append(alphas, np.inf)

        z_plus = alphas[y] - X.dot(beta)
        z_minus = alphas[y-1] - X.dot(beta)
        return -2.0 * np.sum(np.log(logistic(z_plus) - logistic(z_minus))) 