import numpy as np
import sklearn.metrics.pairwise as sk


def hsic_approx(X, y, L, L_ones, feature_kernel=sk.rbf_kernel, label_kernel=sk.rbf_kernel, gamma=1. / 12):
    """Compute an approximation of the Hilbert-Schmidt Indenpendence Criterion (HSIC_1)"""
    K = feature_kernel(X, X, gamma) if gamma is not None else feature_kernel(X, X)
    np.fill_diagonal(K, 0)
    m = X.shape[0]

    oneK = np.ones(m).dot(K)
    trKL = np.multiply(K, L.T).sum()

    hsic = 1. / (m * (m - 3)) * (trKL + 1 / ((m - 1) * (m - 2)) *
                                oneK.dot(np.ones(m) * np.ones(m).dot(L_ones)) - 2 / (m - 2) * oneK.dot(L_ones))
    
    return hsic

