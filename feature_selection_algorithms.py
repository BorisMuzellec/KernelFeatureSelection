#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:30:41 2017

@author: boris
"""
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

from copula_dependence import approx_copula, dependency_measure
from kernels import binary_clf_kernel

# There's probably room for optimization


def incremental_selection(X, Y, k, method='copula', kernel=sk.rbf_kernel, gamma=1. / 12):
    """
    Returns all the subsets of features of length smaller than k selected by incremental search
    """
    S = []
    subsets = []
    m = X.shape[1]

    X_ = np.c_[X, Y]
    Z = approx_copula(X_)

    for i in trange(k, leave=False):
        best_score = -1E3
        best_feature = -1
        if i > 0:
            for j in tqdm(set(np.arange(m)) - set(S), leave=False):
                score = 0
                for s in S:
                    score += dependency_measure(Z[:, (j, s)],
                                                kernel, gamma=gamma)
                    #print('dependency between %u and %u: %f' %(j,s,dependency_measure(Z[:, (j, s)], kernel, gamma)))
                score = dependency_measure(
                    Z[:, (j, -1)], kernel, gamma=gamma) - score / i
                #print('dependency between %u and label: %f' %(j, dependency_measure(Z[:, (j, -1)], kernel, gamma)))

                if score > best_score:
                    best_score = score
                    best_feature = j
        else:
            for j in range(m):
                score = dependency_measure(Z[:, (j, -1)], kernel, gamma=gamma)
                if score > best_score:
                    best_score = score
                    best_feature = j

        print('best_feature: %u , best_score: %f' % (best_feature, best_score))
        S.append(best_feature)
        subsets.append(copy.deepcopy(S))

    return subsets


def selection_heuristic(X, Y, k, classifier, method='copula', kernel=sk.rbf_kernel, gamma=1. / 12, cv=10):
    """
    The selection heuristic from Peng and al.
    - use incremental selection to find n sequential feature sets (n large)
    - compare the feature sets usign cross validation to find a range k of small error (mean + variance)
    - take the smallest set with smallest error
    """

    print("Performing incremental selection")
    subsets = incremental_selection(
        X, Y, k, method=method, kernel=kernel, gamma=gamma)
    # Store the 95% confidence interval of the cv_score as [lower_bound,
    # upper_bound]
    cv_scores = np.zeros((k, 2))

    print("Computing CV scores")
    for i in trange(k, leave=False):
        scores = cross_val_score(
            classifier, X[:, subsets[i]], y, cv=cv, scoring='neg_mean_squared_error')
        # epsilon = 1 if loss else -1
        # cv_scores[i] = (epsilon * scores.mean() - 0.2 * scores.std(), epsilon * scores.mean() + 0.2 * scores.std())
        cv_scores[i, :] = np.array([scores.mean(), scores.std()])

    # print("Find best score, and undistinguishable scores")
    # # find the highest upper confidence interval bound, then its lower bound,
    # # and all the intervals with upper bound higher than this lower bound
    # score_intervals = cv_scores.items()
    # #[a for(a,s) in sorted(initial_ranking, key=operator.itemgetter(1), reverse=True)][:10]
    # upper_bounds = [u for (s, (l, u)) in score_intervals]
    # best_set_idx = np.argmax(upper_bounds)
    # best_set_lower_bound = [l for (s, (l, u)) in score_intervals][best_set_idx]

    # best_sets = [s for (s, (l, u)) in score_intervals if u >=
        # best_set_lower_bound]
    # print(best_sets)

    # Select the smallest mean errors
    print(cv_scores)
    #smallest_best_set = np.argmin(cv_scores[:, 0] ** 2 + cv_scores[:, 1])
    smallest_best_set = np.argmax(cv_scores[:, 0])

    # Take the smallest best set
    # TODO: implement a clever way of breaking ties
    # set_lengths = [len(subsets[s]) for s in best_sets]
    # smallest_best_set = best_sets[np.argmin(set_lengths)]

    return subsets[smallest_best_set], cv_scores[smallest_best_set]


def hsic_approx(X, y, feat_kernel=sk.rbf_kernel, sigma=1.0, label_kernel=binary_clf_kernel):
    """Compute an approximation of the Hilbert-Schmidt Independence Criterion (HSIC1)"""
    # TODO: use sigma parameter
    K = feat_kernel(X, X)
    L = label_kernel(y, y)
    m = X.shape[0]

    for i in range(m):
        K[i, i] = 0
        L[i, i] = 0
    
    oneK = np.ones(m).dot(K)
    Lone = L.dot(np.ones(m))
    trKL = np.multiply(K, L.T).sum()
    hsic = 1 / (m * (m - 3)) * (trKL + 1 / ((m - 1) * (m - 2)) * oneK.dot(np.ones(m) * np.ones(m).dot(Lone)) - 2 / (m - 2) * oneK.dot(Lone))
    return hsic


def bahsic_selection(X, y, feat_kernel=sk.rbf_kernel, sigma=1.0, label_kernel=binary_clf_kernel):
    """Implement Backward Elimination using Hilbert-Schmidt Independence Criterion
        Reference: "Feature Selection via Dependence Maximization", §4.1, Le Sing, Smola, Gretton, Bedo, Borgwardt

        Input:
            X: dataset features
            y: dataset labels
        Output:
            subset of features
    """
    S = set(range(X.shape[1]))
    T = set()
    while len(S) > 0:
        sigma = sigma
        subset_size = int(math.ceil(0.1 * len(S))) 
        if subset_size <= 1:
            return T
        best_hsic_sum = -np.inf
        best_subset = None
        for subset in tqdm(combinations(S, subset_size), total=int(binom(len(S), subset_size)), leave=False):
            subset = set(subset)
            hsic_sum = 0.0
            for j in subset:
                feats = np.array(list(subset - set([j])))
                hsic_sum += hsic_approx(X[:, feats], y, feat_kernel, sigma, label_kernel)
            if hsic_sum > best_hsic_sum:
                best_hsic_sum = hsic_sum
                best_subset = subset
        S = S - best_subset
        T = T.union(best_subset)

    return T




boston = load_boston()
X = boston.data
y = boston.target

print(bahsic_selection(X, y))

#clf = svm.SVC(kernel='rbf', C=1)

params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.1, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)


# print(selection_heuristic(X, y, 12, clf, method='copula',
#                          kernel=sk.rbf_kernel, gamma = 6, cv = 5))
