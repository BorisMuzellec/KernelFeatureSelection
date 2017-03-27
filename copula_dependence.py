#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from math import erf, pi

import numpy as np
import sklearn.metrics.pairwise as sk


def approx_copula(X):
    """
    input:
          X: dataset where each line corresponds to a feature vector
    output:
          Z: copula approximation (Z_i,j = rank of x_i wrt the j-th feature)
    """
    return (np.argsort(X, axis=0) + 1) / X.shape[0]


def erf_aux(Z, gamma):
    (m, d) = Z.shape

    ker = sk.rbf_kernel(Z, Z)
    np.fill_diagonal(ker, 0)
    ker_sum = ker.sum() / (m * (m - 1))

    sum_ = 0
    for i in range(m):
        prod_ = 1
        for j in range(d):
            prod_ *= (np.sqrt(pi) / (2 * gamma)) * \
                (erf(gamma * (1 - Z[i, j])) + erf(gamma * Z[i, j]))
        sum_ += prod_

    integral = np.power(np.sqrt(pi) / gamma * erf(gamma) -
                        (np.exp(-gamma**2) - 1) / gamma**2, d)

    # print Z
    return ker_sum - 2. / m * sum_ + integral


def copula_measure(Z_X, Z_Y, kernel=sk.rbf_kernel, gamma=1. / 12):
    """
    input:
          Z: copula approximation (Z_i,j = rank of x_i wrt the j-th feature)
    output:
          kernel copula independence measure of X (where Z = copula(X))
    """
    
    Z = np.c_[Z_X, Z_Y]
    m = Z.shape[0]
    n = 5 * m

    if kernel == sk.rbf_kernel:
        I = erf_aux(Z, gamma)
    else:
        U = np.random.uniform(size=(n, Z.shape[1]))
        I = kernel(Z, Z).mean() + kernel(U, U).mean() - 2 * kernel(Z, U).mean()

    return np.sqrt(I)


def mRMR(X, Y, S, kernel=sk.rbf_kernel):
    """
    Max Relevance - Min Redundancy kernel copula independence measure
    input:
        X: feature dataset
        Y: labels
        S: index of features to test
    output:
        Max-Relevance Min-Redundancy measure of the sample
    """
    n = len(S)

    X_ = np.c_[X, Y]
    Z = approx_copula(X_)

    maxRelevance = 0
    minRedundancy = 0
    for i in S:
        maxRelevance += copula_measure(Z[:, (i, -1)], kernel)
        for j in S:
            minRedundancy += copula_measure(Z[:, (i, j)], kernel)
    maxRelevance /= n
    minRedundancy /= n**2

    return maxRelevance - minRedundancy
