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
from sklearn.model_selection import cross_val_score
from tqdm import tqdm, trange

from copula_dependence import approx_copula


def incremental_selection(X, Y, k, measure):
    """
    Returns all the subsets of features of length smaller than k selected by incremental search
    """
    S = []
    subsets = []
    m = X.shape[1]

    X_ = np.c_[X, Y]
    Z = approx_copula(X_) if measure.measure == 'copula' else X_

    for i in trange(k, leave=False):
        best_score = -1E3
        best_feature = -1
        if i > 0:
            for j in tqdm(set(np.arange(m)) - set(S), leave=False):
                score = 0
                for s in S:
                    score += measure.score(Z[:, j], Z[:, s])
                    #print('dependency between %u and %u: %f' %(j,s,copula_measure(Z[:, (j, s)], kernel, gamma)))
                score = measure.score(Z[:, j, ], Z[:, -1]) - score / i
                #print('dependency between %u and label: %f' %(j, copula_measure(Z[:, (j, -1)], kernel, gamma)))

                if score > best_score:
                    best_score = score
                    best_feature = j
        else:
            for j in range(m):
                score = measure.score(Z[:, j, ], Z[:, -1])
                if score > best_score:
                    best_score = score
                    best_feature = j

        # print('best_feature: %u , best_score: %f' % (best_feature, best_score))
        S.append(best_feature)
        subsets.append(copy.deepcopy(S))

    return subsets


def selection_heuristic(X, Y, k, classifier, measure, cv=10, regression = True):
    """
    The selection heuristic from Peng and al.
    - use incremental selection to find n sequential feature sets (n large)
    - compare the feature sets usign cross validation to find a range k of small error (mean + variance)
    - take the smallest set with smallest error
    """

    print("Performing incremental selection\n")
    subsets = incremental_selection(
        X, Y, k, measure=measure)
    cv_scores = np.zeros((k, 2))

    print("Computing CV scores\n")
    for i in trange(k, leave=False):
        scores = cross_val_score(
            classifier, X[:, subsets[i]], Y, cv=cv, scoring='neg_mean_squared_error' if regression else 'accuracy')
        # cv_scores[i] = (epsilon * scores.mean() - 0.2 * scores.std(), epsilon * scores.mean() + 0.2 * scores.std())
        cv_scores[i, :] = np.array([scores.mean(), scores.std()])

    # Select the smallest mean errors
    #print(cv_scores)
    #smallest_best_set = np.argmin(cv_scores[:, 0] ** 2 + cv_scores[:, 1])
    smallest_best_set = np.argmax(cv_scores[:, 0])

    return subsets[smallest_best_set], cv_scores[smallest_best_set]


def backward_selection(X, y, t, measure, classifier = None, cv=10, regression = True):
    """Implements Backward Elimination
        Reference: "Feature Selection via Dependence Maximization", §4.1, Le Sing, Smola, Gretton, Bedo, Borgwardt

        Input:
            X: dataset features
            y: dataset labels
            t: desired number of features
            measure: dependency measure
        Output:
            subset of features of size t

    """
    S = set(range(X.shape[1]))
    T = list()
    
    if measure.measure == 'copula':
          X = approx_copula(X) 
          y = approx_copula(y)
    
    while len(S) > 1:
        subset_size = int(math.ceil(0.1 * len(S)))
        best_score_sum = -np.inf
        best_subset = None
        for subset in tqdm(combinations(S, subset_size), total=int(binom(len(S), subset_size)), leave=False):
            subset = set(subset)
            score_sum = 0.0
            for j in subset:
                feats = np.array(list(S - set([j])))
                score_sum += measure.score(X[:, feats], y)
            if score_sum > best_score_sum:
                best_score_sum = score_sum
                best_subset = subset
        S = S - best_subset
        T = T + list(best_subset)
              
        scores = []
        if classifier is not None:
              scores = cross_val_score(
            classifier, X[:,(T + list(S))[:-t]], y, cv=cv, scoring='neg_mean_squared_error' if regression else 'accuracy')
              

    return (T + list(S))[-t:], scores.mean(), scores.std()


def forward_selection(X, y, t, measure, classifier = None, cv=10, regression = True):
    """Implements Forward Selection
        Reference: "Feature Selection via Dependence Maximization", §4.2, Le sing, Smola, Gretton, Bedo, Borgwardt

        Input:
            X: dataset features
            y: dataset labels
            t: desired number of output features
            measure: dependency measure
        Output:
            subset of features of size t
            if classifier not None: the cv score of the classifier on the chosen subset
                  
    """
    S = set(range(X.shape[1]))
    T = list()
    
    if measure.measure == 'copula':
          X = approx_copula(X) 
          y = approx_copula(y)
          
    while len(S) > 1:
        if len(T) > t:
            return T[:t]
        subset_size = int(math.ceil(0.1 * len(S)))
        best_score_sum = -np.inf
        best_subset = None
        for subset in tqdm(combinations(S, subset_size), total=int(binom(len(S), subset_size)), leave=False):
            subset = set(subset)
            score_sum = 0.0
            for j in subset:
                feats = np.array(T + [j])
                score_sum += measure.score(X[:, feats], y)
            if score_sum > best_score_sum:
                best_score_sum = score_sum
                best_subset = subset
        S = S - best_subset
        T = T + list(best_subset)
        
        scores = []
        if classifier is not None:
              scores = cross_val_score(
            classifier, X[:,(T + list(S))[:t]], y, cv=cv, scoring='neg_mean_squared_error' if regression else 'accuracy')
              

    return (T + list(S))[:t], scores.mean(), scores.std()


