#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:30:41 2017

@author: boris
"""
import copy

import numpy as np
import sklearn.metrics.pairwise as sk
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import ensemble
from tqdm import tqdm, trange

from copula_dependence import approx_copula, dependency_measure

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


boston = load_boston()
X = boston.data
y = boston.target

print(incremental_selection(X, y, 1, gamma=6))

#clf = svm.SVC(kernel='rbf', C=1)

params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.1, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)


# print(selection_heuristic(X, y, 12, clf, method='copula',
#                          kernel=sk.rbf_kernel, gamma = 6, cv = 5))
