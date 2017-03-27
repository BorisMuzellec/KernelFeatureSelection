#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:30:41 2017

@author: boris
"""
from __future__ import division


import copy
from itertools import combinations
import math

import numpy as np
from scipy.special import binom
import sklearn.metrics.pairwise as sk
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import ensemble
from tqdm import tqdm, trange

from measure import Dependency_Measure
from kernels import binary_clf_kernel
from copula_dependence import approx_copula
# There's probably room for optimization


def incremental_selection(X, Y, k, measure):
    """
    Returns all the subsets of features of length smaller than k selected by incremental search
    """
    S = []
    subsets = []
    m = X.shape[1]

    X_ = np.c_[X, Y]
    Z = approx_copula(X_) if measure.measure =='copula' else X_

    for i in trange(k, leave=False):
        best_score = -1E3
        best_feature = -1
        if i > 0:
            for j in tqdm(set(np.arange(m)) - set(S), leave=False):
                score = 0
                for s in S:
                    score += measure.score(Z[:, j],Z[:, s])
                    #print('dependency between %u and %u: %f' %(j,s,copula_measure(Z[:, (j, s)], kernel, gamma)))
                score = measure.score(Z[:, j,], Z[:, -1]) - score / i
                #print('dependency between %u and label: %f' %(j, copula_measure(Z[:, (j, -1)], kernel, gamma)))

                if score > best_score:
                    best_score = score
                    best_feature = j
        else:
            for j in range(m):
                score = measure.score(Z[:, j,], Z[:, -1]) 
                if score > best_score:
                    best_score = score
                    best_feature = j

        print('best_feature: %u , best_score: %f' % (best_feature, best_score))
        S.append(best_feature)
        subsets.append(copy.deepcopy(S))

    return subsets


def selection_heuristic(X, Y, k, classifier, measure, cv=10):
    """
    The selection heuristic from Peng and al.
    - use incremental selection to find n sequential feature sets (n large)
    - compare the feature sets usign cross validation to find a range k of small error (mean + variance)
    - take the smallest set with smallest error
    """

    print("Performing incremental selection")
    subsets = incremental_selection(
        X, Y, k, measure= measure)
    cv_scores = np.zeros((k, 2))

    print("Computing CV scores")
    for i in trange(k, leave=False):
        scores = cross_val_score(
            classifier, X[:, subsets[i]], y, cv=cv, scoring='neg_mean_squared_error')
        # cv_scores[i] = (epsilon * scores.mean() - 0.2 * scores.std(), epsilon * scores.mean() + 0.2 * scores.std())
        cv_scores[i, :] = np.array([scores.mean(), scores.std()])


    # Select the smallest mean errors
    print(cv_scores)
    #smallest_best_set = np.argmin(cv_scores[:, 0] ** 2 + cv_scores[:, 1])
    smallest_best_set = np.argmax(cv_scores[:, 0])


    return subsets[smallest_best_set], cv_scores[smallest_best_set]


def bahsic_selection(X, y, t, measure):
    """Implement Backward Elimination using Hilbert-Schmidt Independence Criterion
        Reference: "Feature Selection via Dependence Maximization", §4.1, Le Sing, Smola, Gretton, Bedo, Borgwardt

        Input:
            X: dataset features
            y: dataset labels
            t: desired number of features
        Output:
            subset of features of size t
            
        WARNING: does not support copula yet
    """
    S = set(range(X.shape[1]))
    T = list()
    while len(S) > 1:
        subset_size = int(math.ceil(0.1 * len(S)))
        best_hsic_sum = -np.inf
        best_subset = None
        for subset in tqdm(combinations(S, subset_size), total=int(binom(len(S), subset_size)), leave=False):
            subset = set(subset)
            hsic_sum = 0.0
            for j in subset:
                feats = np.array(list(S - set([j])))
                hsic_sum += measure.score(X[:, feats], y)
            if hsic_sum > best_hsic_sum:
                best_hsic_sum = hsic_sum
                best_subset = subset
        S = S - best_subset
        T = T + list(best_subset)

    return (T + list(S))[-t:]


def fohsic_selection(X, y, t, measure):
    """Implement Forward Selection using Hilbert-Schmidt Independence Criterion
        Reference: "Feature Selection via Dependence Maximization", §4.2, Le sing, Smola, Gretton, Bedo, Borgwardt

        Input:
            X: dataset features
            y: dataset labels
            t: desired number of output features
        Output:
            subset of features of size t
            
      WARNING: does not support copula yet
    """
    S = set(range(X.shape[1]))
    T = list()
    while len(S) > 1:
        subset_size = int(math.ceil(0.1 * len(S)))
        best_hsic_sum = -np.inf
        best_subset = None
        for subset in tqdm(combinations(S, subset_size), total=int(binom(len(S), subset_size)), leave=False):
            subset = set(subset)
            hsic_sum = 0.0
            for j in subset:
                feats = np.array(T + [j])
                hsic_sum += measure.score(X[:, feats], y)
            if hsic_sum > best_hsic_sum:
                best_hsic_sum = hsic_sum
                best_subset = subset
        S = S - best_subset
        T = T + list(best_subset)
    
    return (T + list(S))[:t]




boston = load_boston()
X = boston.data
y = boston.target


HSIC = Dependency_Measure(measure = 'hsic', feature_kernel = sk.rbf_kernel, label_kernel= sk.rbf_kernel, gamma=1./12)
COPULA = Dependency_Measure(measure = 'copula', feature_kernel = sk.rbf_kernel,  gamma= 6)

#print(bahsic_selection(X, y, 4))

#clf = svm.SVC(kernel='rbf', C=1)

params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.1, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)


print(selection_heuristic(X, y, 6, clf, measure = COPULA))
