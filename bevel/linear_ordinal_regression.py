import numpy as np
import warnings

from numdifftools import Jacobian
from numpy.linalg import inv
from pandas import DataFrame
from scipy import optimize
from scipy.linalg import block_diag
from scipy.special import expit
from scipy.stats import norm
from scipy.stats import kendalltau


__all__ = ['OrderedLogit', 'OrderedProbit']


class LinearOrdinalRegression():
    """
    A general class for linear ordinal regression fitting. The cumulative distribution
    for the probability of being classified into category p depends linearly on the regressors
    through a link function Phi:

    P(Y < p | X_i) = Phi(alpha_p - X_i.beta)

    Parameters:
      link: a link function that is increasing and bounded by 0 and 1
      deriv_link: the derivative of the link function
      significance: the significance of confidence levels reported in the fit summary
    """

    def __init__(self, link, deriv_link, significance=0.95):
        self.significance = significance
        self.link = link
        self.deriv_link = deriv_link

    def fit(self, X, y, maxfun=100000, maxiter=100000, epsilon=10E-9):
        """
        Fit a linear ordinal regression model to the input data by maximizing the
        log likelihood function.

        Parameters:
          X: a pandas DataFrame or numpy array of numerical regressors
          y: a column of ordinal-valued data
          maxfun: the maximum number of function calls used by scipy.optimize()
          maxiter: the maximum number of iterations used by scipy.optimize()
          epsilon: the minimum difference between successive intercepts, alpha_{p+1} - alpha_p

        Returns:
          self, with alpha_, beta_, coef_, se_, p_values_ and score_ properties determined
        """

        X_data, X_scale, X_mean, X_std = self._prepare_X(X)
        y_data = self._prepare_y(y)

        beta_guess = np.zeros(self.n_attributes)
        gamma_guess = np.ones(self.n_classes - 1)
        bounds = [(None, None)] * (self.n_attributes + 1) + [(epsilon, None)] * (self.n_classes - 2)

        optimization = optimize.minimize(
            self._log_likelihood,
            np.append(beta_guess, gamma_guess),
            jac=self._gradient,
            args=(X_scale, y_data),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxfun': maxfun, 'maxiter': maxiter}
        )

        if not optimization.success:
            message = 'Likelihood maximization failed - ' + str(optimization.message, 'utf-8')
            warnings.warn(message, RuntimeWarning)

        self.beta_ = optimization.x[:self.n_attributes] / X_std
        gamma = optimization.x[self.n_attributes:]
        gamma[0] = gamma[0] + X_mean.dot(self.beta_)
        self.alpha_ = np.cumsum(gamma)

        self.se_ = self._compute_standard_errors(np.append(self.beta_, gamma), X_data, y_data)
        self.p_values_ = self._compute_p_values()
        self.score_ = self._compute_score(X_data, y_data)

        return self

    @property
    def coef_(self):
        return np.append(self.beta_, self.alpha_)

    @property
    def summary(self):
        """
        Summary statistics describing the fit.

        Returns:
          a pandas DataFrame with columns coef, se(coef), p, lower, upper
        """

        significance_std_normal = norm.ppf((1. + self.significance) / 2.)
        df = self.attribute_names.set_index('attribute names')

        df['beta'] = self.beta_
        df['se(beta)'] = self.se_[:self.n_attributes]
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

    def predict_linear_product(self, X):
        """
        Predict the linear product score X.beta for a set of input variables

        Parameters:
          X: a pandas DataFrame or numpy array of inputs to predict, one row per input

        Returns:
          a numpy array with the predicted linear product score for each input
        """

        if X.ndim == 1:
            X = X[None, :]
        return X.dot(self.beta_)[:, None]

    def predict_probabilities(self, X):
        """
        Predict the probability of input variables belonging to each ordinal class

        Parameters:
          X: a pandas DataFrame or numpy array of inputs to predict, one row per input

        Returns:
          a numpy array with n_classes columns listing the probability of belonging to each class
        """

        bounded_alpha = self._bounded_alpha(self.alpha_)
        z = bounded_alpha - self.predict_linear_product(X)
        cumulative_dist = self.link(z)
        return np.diff(cumulative_dist)

    def predict_class(self, X):
        """
        Predict the most likely class for a set of input variables

        Parameters:
          X: a pandas DataFrame or numpy array of inputs to predict, one row per input

        Returns:
          a numpy array with the predicted most likely class for each input
        """

        probs = self.predict_probabilities(X)
        raw_predictions = np.argmax(probs, axis=1) + 1
        return np.vectorize(self._y_dict.get)(raw_predictions)

    def _prepare_X(self, X):
        X_data = np.asarray(X)
        X_data = X_data[:, None] if len(X_data.shape) == 1 else X_data
        self.N, self.n_attributes = X_data.shape
        self.attribute_names = self._get_column_names(X)
        X_std = X_data.std(0)
        X_mean = X_data.mean(0)

        trivial_X = X_std == 0
        if any(trivial_X):
            raise ValueError(
                'The regressors {} have 0 variance.'.format(self.attribute_names[trivial_X].values)
            )

        return X_data, (X_data - X_mean) / X_std, X_mean, X_std

    def _prepare_y(self, y):
        y_data = np.asarray(y).astype(int)
        y_values = np.sort(np.unique(y_data))

        self.n_classes = len(y_values)
        y_range = np.arange(1, self.n_classes + 1)
        self._y_dict = dict(zip(y_range, y_values))

        y_data = np.vectorize(dict(zip(y_values, y_range)).get)(y_data)

        self._indicator_plus = np.array([y_data == i + 1 for i in range(self.n_classes - 1)]) * 1.0
        self._indicator_minus = np.array([y_data - 1 == i + 1 for i in range(self.n_classes - 1)]) * 1.0

        return y_data

    def _get_column_names(self, X):
        if isinstance(X, DataFrame):
            column_names = X.columns.tolist()
        else:
            column_names = ['column_' + str(i+1) for i in range(self.n_attributes)]
        return DataFrame(column_names, columns=['attribute names'])

    def _log_likelihood(self, coefficients, X_data, y_data):
        beta = coefficients[:self.n_attributes]
        gamma = coefficients[self.n_attributes:]
        bounded_alpha = self._bounded_alpha(np.cumsum(gamma))
        z_plus = bounded_alpha[y_data] - X_data.dot(beta)
        z_minus = bounded_alpha[y_data-1] - X_data.dot(beta)
        return - 1.0 * np.sum(np.log(self.link(z_plus) - self.link(z_minus)))

    def _gradient(self, coefficients, X_data, y_data):
        beta = coefficients[:self.n_attributes]
        gamma = coefficients[self.n_attributes:]
        bounded_alpha = self._bounded_alpha(np.cumsum(gamma))

        deriv_link_plus = self.deriv_link(bounded_alpha[y_data] - X_data.dot(beta))
        deriv_link_minus = self.deriv_link(bounded_alpha[y_data-1] - X_data.dot(beta))
        denominator = self.link(bounded_alpha[y_data] - X_data.dot(beta)) - self.link(bounded_alpha[y_data-1] - X_data.dot(beta))

        #the only way the denominator can vanish is if the numerator also vanishes
        #so we can safely overwrite any division by zero that arises numerically
        denominator[denominator == 0] = 1

        quotient_plus = deriv_link_plus / denominator
        quotient_minus = deriv_link_minus / denominator

        alpha_gradient = (quotient_plus - quotient_minus).dot(X_data)
        beta_gradient = self._indicator_minus.dot(quotient_minus) - self._indicator_plus.dot(quotient_plus)

        return np.append(alpha_gradient, beta_gradient).dot(self._compute_basis_change())

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


class OrderedLogit(LinearOrdinalRegression):
    """
    This class implements ordinal logistic regression fitting. The link function in this
    case is the logistic function and the cumulative distribution is parameterized as follows:

    P(Y < p | X_i) = 1 / 1 + exp(X_i.beta - alpha_p)

    Parameters:
      significance: the significance of confidence levels reported in the fit summary
    """

    @staticmethod
    def diff_expit(z):
        return expit(z) * (1 - expit(z))

    def __init__(self, significance=0.95):
        super().__init__(expit, self.diff_expit, significance=significance)


class OrderedProbit(LinearOrdinalRegression):
    """
    This class implements ordered probit regression fitting. The link function in this
    case is the logistic function and the cumulative distribution is parameterized as follows:

    P(Y < p | X_i) = Probit(alpha_p - X_i.beta)

    Parameters:
      significance: the significance of confidence levels reported in the fit summary
    """

    def __init__(self, significance=0.95):
        super().__init__(norm.cdf, norm.pdf, significance=significance)

