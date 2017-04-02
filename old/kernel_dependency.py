#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:13:18 2017

@author: boris
"""

# Copula-free kernel dependency
import numpy as np
import sklearn.metrics.pairwise as sk


# The unbiased estimator
#def hsic(X, Y, feature_kernel=sk.rbf_kernel, label_kernel=sk.rbf_kernel, gamma = 1./12):
#    """
#    Allows to use different kernels for features and labels, which is usefull in the case of classification
#    For regression, linear or rbf kernels work well
#    """
#    m = X.shape[0]
#
#    K = feature_kernel(X, X, gamma)
#    L = label_kernel(Y, Y)
#
#    np.fill_diagonal(K, 0)
#    np.fill_diagonal(L, 0)
#
#    KL = K.dot(L)
#    return (np.trace(KL) + K.sum() * L.sum() / ((m - 1) * (m - 2)) - 2. / (m - 2) * KL.sum()) / (m * (m - 3))
#
#    
def hsic_approx(X, y, L, Lones, feature_kernel=sk.rbf_kernel, label_kernel = sk.rbf_kernel, gamma = 1./12):
    """Compute an approximation of the Hilbert-Schmidt Independence Criterion (HSIC1)"""
    K = feature_kernel(X, X, gamma) if gamma is not None else feature_kernel(X,X)
    # L = label_kernel(y, y)
    m = X.shape[0]
    
    for i in range(m):
        K[i, i] = 0
        # L[i, i] = 0

    oneK = np.ones(m).dot(K)
    # Lones = L.dot(np.ones(m))
    trKL = np.multiply(K, L.T).sum()
    hsic = 1. / (m * (m - 3)) * (trKL + 1 / ((m - 1) * (m - 2)) *
                                oneK.dot(np.ones(m) * np.ones(m).dot(Lones)) - 2 / (m - 2) * oneK.dot(Lones))
    return hsic
