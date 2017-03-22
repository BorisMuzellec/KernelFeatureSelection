#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.metrics.pairwise as sk
from sklearn.datasets import load_iris


def approx_copula(X):
    """
    input:
          X: dataset where each line corresponds to a feature vector
    output:
          Z: copula approximation (Z_i,j = rank of x_i wrt the j-th feature)
    """
    return (np.argsort(X, axis=1) + 1) / X.shape[0]


def independence_measure(Z, kernel=sk.rbf_kernel):
    """
    input:
          Z: copula approximation (Z_i,j = rank of x_i wrt the j-th feature)
    output:
          kernel copula independence measure of X (where Z = copula(X))
    """
    m = Z.shape[0]
    n = 5 * m
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
        maxRelevance += independence_measure(Z[:, (i, -1)], kernel)
        for j in S:
            minRedundancy += independence_measure(Z[:, (i, j)], kernel)
    maxRelevance /= n
    minRedundancy /= n**2

    return maxRelevance - minRedundancy


iris = load_iris()
X = iris.data
y = iris.target
