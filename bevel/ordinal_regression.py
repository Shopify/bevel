import numpy as np
import pandas as pd
import warnings

from numdifftools import Hessian
from numpy.linalg import inv
from scipy import optimize
from scipy.linalg import block_diag
from scipy.stats import norm



def logistic(z):
    positive_z = z > 0
    logistic = np.zeros_like(z, dtype=np.float)
    logistic[positive_z] = 1.0 / (1 + np.exp(-z[positive_z]))
    exp_z = np.exp(z[~positive_z])
    logistic[~positive_z] = exp_z / (1.0 + exp_z)
    return logistic


class OrdinalRegression():

    def __init__(self, significance=0.95, maxfun=100000, maxiter=100000):
        self.significance = significance
        self.maxfun = maxfun
        self.maxiter = maxiter

    def fit(self, X, y):
        X_data, y_data = self._prepare(X, y)

        beta_guess = np.zeros(self.n_attributes)
        gamma_guess = np.ones(self.n_classes - 1)
        bounds = [(None, None)] * (self.n_attributes + 1) + [(0, None)] * (self.n_classes - 2)

        optimization = optimize.minimize(
            self._log_likelihood,
            np.append(beta_guess, gamma_guess),
            args=(X_data, y_data),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxfun': self.maxfun, 'maxiter': self.maxiter}
        )

        if not optimization.success:
            warning_message = 'Likelihood maximization failed - ' + str(optimization.message, 'utf-8')
            warnings.warn(warning_message, RuntimeWarning)

        self._set_coefficients(optimization.x)
        self.se_ = self._compute_standard_errors(X_data, y_data)
        self.p_values_ = self._compute_p_values()
        return self

    @property
    def summary(self):
        """
        Summary statistics describing the fit.

        Returns
        -------
        df : pd.DataFrame
            Contains columns coef, se(coef), p, lower, upper
        """

        significance_std_normal = norm.ppf((1. + self.significance) / 2.)
        df = pd.DataFrame(index=self.attribute_names)
        df['coef'] = self.beta_
        df['se(coef)'] = self.se_[:self.n_attributes]
        df['p'] = self.p_values_[:self.n_attributes]
        conf_interval = significance_std_normal * self.se_[:self.n_attributes]
        df['lower %.2f' % self.significance] = self.beta_ - conf_interval
        df['upper %.2f' % self.significance] = self.beta_ + conf_interval
        return df

    def print_summary(self):
        """
        Print summary statistics describing the fit.

        """
        def significance_code(p):
            if p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            elif p < 0.1:
                return '.'
            else:
                return ' '

        df = self.summary
        # Significance codes last
        df[''] = [significance_code(p) for p in df['p']]

        # Print information about data first
        print('n={}'.format(self.N), end="\n")
        print(df.to_string(float_format=lambda f: '{:4.4f}'.format(f)))
        # Significance code explanation
        print('---')
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 ",
              end='\n\n')
        return

    def predict(self, X):
        if X.ndim == 1:
            X = X[None, :]

        bounded_alpha = self._bounded_alpha(self.alpha_)
        z = (bounded_alpha - np.dot(X, self.beta_[:, None]))
        cumulative_dist = logistic(z)
        raw_predictions = np.argmax(np.diff(cumulative_dist), axis=1) + 1
        return np.vectorize(self.inverse_y_dict.get)(raw_predictions)

    def _prepare(self, X, y):
        X_data = np.asarray(X)
        self.N, self.n_attributes = X_data.shape
        if isinstance(X, pd.DataFrame):
            self.attribute_names = X.columns.tolist()
        else:
            self.attribute_names = ['column_' + str(i) for i in range(self.n_attributes)]

        y_data = np.asarray(y).astype(np.int)
        y_values = np.sort(np.unique(y_data))
        self.n_classes = len(y_values)
        y_range = np.arange(1, self.n_classes + 1)
        self.y_dict = dict(zip(y_values, y_range))
        self.inverse_y_dict = dict(zip(y_range, y_values))
        y_data = np.vectorize(self.y_dict.get)(y_data)

        return X_data, y_data

    def _log_likelihood(self, coefficients, X_data, y_data):
        beta = coefficients[:self.n_attributes]
        gamma = coefficients[self.n_attributes:]
        bounded_alpha = self._bounded_alpha(np.cumsum(gamma))
        z_plus = bounded_alpha[y_data] - X_data.dot(beta)
        z_minus = bounded_alpha[y_data-1] - X_data.dot(beta)
        return - 1.0 * np.sum(np.log(logistic(z_plus) - logistic(z_minus)))

    def _set_coefficients(self, optimization_x):
        self.gamma_ = optimization_x[self.n_attributes:]
        self.beta_ = optimization_x[:self.n_attributes]
        self.alpha_ = np.cumsum(self.gamma_)
        self.coef_ = np.append(self.beta_, self.alpha_)

    def _compute_standard_errors(self, X_data, y_data):
        hessian_function = Hessian(self._log_likelihood, method='forward')
        H = hessian_function(np.append(self.beta_, self.gamma_), X_data, y_data)
        J = self._compute_jacobian()
        return np.sqrt(np.diagonal(J.dot(inv(H)).dot(J.T)))

    def _compute_jacobian(self):
        upper_left_diagonal = np.identity(self.n_attributes)
        lower_right_triangular = np.tril(np.ones((self.n_classes - 1, self.n_classes - 1)))
        return block_diag(upper_left_diagonal, lower_right_triangular)

    def _compute_z_values(self):
        return self.coef_ / self.se_

    def _compute_p_values(self):
        z_magnitudes = np.abs(self._compute_z_values())
        p_values = 1 - norm.cdf(z_magnitudes) + norm.cdf(-z_magnitudes)
        return p_values

    @staticmethod
    def _bounded_alpha(alpha):
        _alpha = np.insert(alpha, 0, -np.inf)
        _alpha = np.append(_alpha, np.inf)
        return _alpha