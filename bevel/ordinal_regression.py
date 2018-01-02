import numpy as np
import pandas as pd
import warnings

from numdifftools import Jacobian
from numpy.linalg import inv
from scipy import optimize
from scipy.linalg import block_diag
from scipy.stats import norm
from scipy.stats import kendalltau




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
        X_data, X_std = self._prepare_X(X)
        y_data = self._prepare_y(y)

        beta_guess = np.zeros(self.n_attributes)
        gamma_guess = np.ones(self.n_classes - 1)
        bounds = [(None, None)] * (self.n_attributes + 1) + [(0, None)] * (self.n_classes - 2)

        optimization = optimize.minimize(
            self._log_likelihood,
            np.append(beta_guess, gamma_guess),
            jac=self._gradient,
            args=(X_data, y_data),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxfun': self.maxfun, 'maxiter': self.maxiter}
        )

        if not optimization.success:
            warning_message = 'Likelihood maximization failed - ' + str(optimization.message, 'utf-8')
            warnings.warn(warning_message, RuntimeWarning)

        self.se_ = self._compute_standard_errors(optimization.x, X_data, y_data)
        self.se_[:self.n_attributes] = self.se_[:self.n_attributes] / X_std
        
        self.alpha_ = np.cumsum(optimization.x[self.n_attributes:])
        self.beta_ = optimization.x[:self.n_attributes] / X_std
        
        self.coef_ = np.append(self.beta_, self.alpha_)
        self.p_values_ = self._compute_p_values()
        self.score_ = self._compute_score(X_data, y_data)

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
        print("Somers' D = {:.3f}".format(self.score_))
        return

    def predict_probabilities(self, X):
        if X.ndim == 1:
            X = X[None, :]

        bounded_alpha = self._bounded_alpha(self.alpha_)
        z = bounded_alpha - X.dot(self.beta_)[:, None]
        cumulative_dist = logistic(z)
        return np.diff(cumulative_dist)

    def predict_class(self, X):
        probs = self.predict_probabilities(X)
        raw_predictions = np.argmax(probs, axis=1) + 1
        return np.vectorize(self._y_dict.get)(raw_predictions)

    def _prepare(self, X, y):
        X_data = np.asarray(X)
        self.N, self.n_attributes = X_data.shape
        if isinstance(X, pd.DataFrame):
            self.attribute_names = X.columns.tolist()
        else:
            self.attribute_names = ['column_' + str(i) for i in range(self.n_attributes)]

    def _prepare_X(self, X):
        X_data = np.asarray(X)
        self.N, self.n_attributes = X_data.shape
        self.attribute_names = self._get_column_names(X)
        
        X_std = X_data.std(0)
        X_std[X_std == 0] = 1
        
        return X_data / X_std, X_std

    def _prepare_y(self, y):
        y_data = np.asarray(y).astype(np.int)
        y_values = np.sort(np.unique(y_data))
        
        self.n_classes = len(y_values)
        y_range = np.arange(1, self.n_classes + 1)
        self._y_dict = dict(zip(y_range, y_values))
        
        return np.vectorize(dict(zip(y_values, y_range)).get)(y_data)

    def _get_column_names(self, X):
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        return ['column_' + str(i+1) for i in range(self.n_attributes)]

    def _log_likelihood(self, coefficients, X_data, y_data):
        beta = coefficients[:self.n_attributes]
        gamma = coefficients[self.n_attributes:]
        bounded_alpha = self._bounded_alpha(np.cumsum(gamma))
        z_plus = bounded_alpha[y_data] - X_data.dot(beta)
        z_minus = bounded_alpha[y_data-1] - X_data.dot(beta)
        return - 1.0 * np.sum(np.log(logistic(z_plus) - logistic(z_minus)))

    def _gradient(self, coefficients, X_data, y_data):
        beta = coefficients[:self.n_attributes]
        gamma = coefficients[self.n_attributes:]
        bounded_alpha = self._bounded_alpha(np.cumsum(gamma))

        phi_plus = logistic(bounded_alpha[y_data] - X_data.dot(beta))
        phi_minus = logistic(bounded_alpha[y_data-1] - X_data.dot(beta))
        quotient_plus = phi_plus * (1 - phi_plus) / (phi_plus - phi_minus)
        quotient_minus = phi_minus * (1 - phi_minus) / (phi_plus - phi_minus)
        indicator_plus = np.array([y_data == i + 1 for i in range(self.n_classes - 1)]) * 1.0
        indicator_minus = np.array([y_data - 1 == i + 1 for i in range(self.n_classes - 1)]) * 1.0

        alpha_gradient = (quotient_plus - quotient_minus).dot(X_data)
        beta_gradient = indicator_minus.dot(quotient_minus) - indicator_plus.dot(quotient_plus)

        return np.append(alpha_gradient, beta_gradient).dot(self._compute_basis_change())

    def _set_coefficients(self, optimization_x):
        self.gamma_ = optimization_x[self.n_attributes:]
        self.beta_ = optimization_x[:self.n_attributes]
        self.alpha_ = np.cumsum(self.gamma_)
        self.coef_ = np.append(self.beta_, self.alpha_)

    def _compute_standard_errors(self, coefficients, X_data, y_data):
        hessian_function = Jacobian(self._gradient, method='forward')
        H = hessian_function(coefficients, X_data, y_data)
        P = self._compute_basis_change()
        return np.sqrt(np.diagonal(P.dot(inv(H)).dot(P.T)))

    def _compute_basis_change(self):
        upper_left_diagonal = np.identity(self.n_attributes)
        lower_right_triangular = np.tril(np.ones((self.n_classes - 1, self.n_classes - 1)))
        return block_diag(upper_left_diagonal, lower_right_triangular)

    def _compute_z_values(self):
        return self.coef_ / self.se_

    def _compute_p_values(self):
        z_magnitudes = np.abs(self._compute_z_values())
        p_values = 1 - norm.cdf(z_magnitudes) + norm.cdf(-z_magnitudes)
        return p_values

    def _compute_score(self, X, y):
        x_beta = X.dot(self.beta_)
        return kendalltau(x_beta, y).correlation

    @staticmethod
    def _bounded_alpha(alpha):
        return np.concatenate(([-np.inf], alpha, [np.inf]))
