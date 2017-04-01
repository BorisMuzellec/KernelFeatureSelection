import numpy as np
import sklearn.metrics.pairwise as sk
from sklearn.metrics import normalized_mutual_info_score

from copula_dependency import copula_measure
from hsic_dependency import hsic_approx


class DependencyMeasure(object):
    """
    A generic class for dependency measures
    """

    def __init__(self, measure='copula', feature_kernel=sk.rbf_kernel, label_kernel=sk.rbf_kernel, gamma=1. / 12):
        """
        Args:
              Measure type (copula, hsic or mutual_information)
              
              Optional arguments
        """

        assert(measure in ['copula', 'hsic', 'mutual_information'])

        self.measure = measure
        self.feature_kernel = feature_kernel
        self.label_kernel = label_kernel
        self.gamma = gamma if feature_kernel != sk.linear_kernel else None
              
        if self.measure == 'copula':
            self.scorer = lambda x, y: copula_measure(x, y, kernel=self.feature_kernel, gamma=self.gamma)
        elif self.measure == 'hsic':
            self.scorer = lambda x, y, l, l_one: hsic_approx(x, y, l, l_one, feature_kernel=self.feature_kernel, label_kernel=self.label_kernel, gamma=self.gamma)
        elif self.measure == 'mutual_information':
            self.scorer = lambda x, y: normalized_mutual_info_score(np.ravel(x), np.ravel(y)) 

    def score(self, X, Y, L, Lones):
        """
        Return the dependency score I(X,Y)
        """
        X_ = X[:, np.newaxis] if len(X.shape) == 1 else X
        Y_ = Y[:, np.newaxis] if len(Y.shape) == 1 else Y
        if self.measure == 'hsic':
            return self.scorer(X_, Y_, L, Lones)
        return self.scorer(X_, Y_)

