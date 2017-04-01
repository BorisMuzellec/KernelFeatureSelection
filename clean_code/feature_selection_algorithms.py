import copy
from itertools import combinations
import math

import numpy as np
from scipy.special import binom
from sklearn.model_selection import cross_val_score
from tqdm import tqdm, trange

from copula_dependency import approx_copula


def backward_selection(X, y, t, measure, estimators, cv=10, regression=True, copula=None):
    """Implement Backward Elimination algorithm for Feature Selection
        Reference: "Feature Selection via Dependence Maximization", §4.1, Le Sing, Smola, Gretton, Bedo, Borgwardt

    :X: dataset features
    :y: dataset labels
    :t: desired number of features
    :measure: dependency measure (instance of DependencyMeasure)
    :estimators: list of estimators used to select the best subset (list of tuples (est_name, est))
    :cv: number of folds for cross-validation
    :regression: boolean, True if the task is a regression, False if it is a classification
    :copula: copula distribution (optional)
    :returns: dict whose keys are estimators names and values are a tuple (best subset, cv mean, cv std)

    """
    S = set(range(X.shape[1]))
    T = list()

    Y = y[:, np.newaxis] if y.ndim == 1 else y

    if measure.measure == 'copula':
        X = approx_copula(X) if copula is None else copula[:, :-1]
        Y = approx_copula(y)

    if measure.measure == 'hsic':
        L = measure.label_kernel(Y, Y)
        np.fill_diagonal(L, 0)
        L_ones = L.dot(np.ones(X.shape[0]))
    else:
        L = None
        L_ones = None

    while len(S) > 1:
        subset_size = int(math.ceil(0.1 * len(S)))
        best_score_sum = -np.inf
        best_subset = None
        for subset in tqdm(combinations(S, subset_size), total=int(binom(len(S), subset_size)), leave=False):
            subset = set(subset)
            score_sum = 0.0
            for j in subset:
                feats = np.array(list(S - set([j])))
                score_sum += measure.score(X[:, feats], Y, L, L_ones)
            if score_sum > best_score_sum:
                best_score_sum = score_sum
                best_subset = subset
        S = S - best_subset
        T = T + list(best_subset)

    T = (T + list(S))[-t:]

    cv_scores = {}
    
    scoring = 'neg_mean_squared_error' if regression else 'accuracy'
    for est_name, est in estimators:
        scores = cross_val_score(est, X[:, T], y, cv=cv, scoring=scoring)
        cv_scores[est_name] = (T, scores.mean(), scores.std())
    
    return cv_scores


def forward_selection(X, y, t, measure, estimators, cv, regression=True, copula=None):
    """Implement Forward Elimination algorithm for Feature Selection
        Reference: "Feature Selection via Dependence Maximization", §4.1, Le Sing, Smola, Gretton, Bedo, Borgwardt

    :X: dataset features
    :y: dataset labels
    :t: desired number of features
    :measure: dependency measure (instance of DependencyMeasure)
    :estimators: list of estimators used to select the best subset (list of tuples (est_name, est))
    :cv: number of folds for cross-validation
    :regression: boolean, True if the task is a regression, False if it is a classification
    :copula: copula distribution (optional)
    :returns: dict whose keys are estimators names and values are a tuple (best subset, cv mean, cv std)

    """
    S = set(range(X.shape[1]))
    T = list()

    Y = y[:, np.newaxis] if y.ndim == 1 else y

    if measure.measure == 'copula':
        X = approx_copula(X) if copula is None else copula[:, :-1]
        Y = approx_copula(y)
    
    if measure.measure == 'hsic':
        L = measure.label_kernel(Y, Y)
        np.fill_diagonal(L, 0)
        L_ones = L.dot(np.ones(X.shape[0]))
    else:
        L = None
        L_ones = None

    while len(S) > 1:
        if len(T) > t:
            T = T[t:]
            break
        subset_size = int(math.ceil(0.1 * len(S)))
        best_score_sum = -np.inf
        best_subset = None
        for subset in tqdm(combinations(S, subset_size), total=int(binom(len(S), subset_size)), leave=False):
            subset = set(subset)
            score_sum = 0.0
            for j in subset:
                feats = np.array(T + [j])
                score_sum += measure.score(X[:, feats], Y, L, L_ones)
            if score_sum > best_score_sum:
                best_score_sum = score_sum
                best_subset = subset
        S = S - best_subset
        T = T + list(best_subset)

    cv_scores = {}
    
    scoring = 'neg_mean_squared_error' if regression else 'accuracy'
    for est_name, est in estimators:
        scores = cross_val_score(est, X[:, T], y, cv=cv, scoring=scoring)
        cv_scores[est_name] = (T, scores.mean(), scores.std())
    
    return cv_scores


def incremental_search(X, y, k, measure, copula=None):
    """Compute all subsets of length less or equal to k, selected by incremental search.
        Reference: "Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy", §2.2, Peng et al.

    :X: dataset features
    :y: dataset labels
    :k: maximum size of subsets
    :measure: dependency measure (instance of DependencyMeasure)
    :copula: copula distribution (optional)
    :returns: list of subsets of sizes 1 to k

    """
    S = []
    subsets = []
    m = X.shape[1]
    Y = y[:, np.newaxis] if y.ndim == 1 else y

    X_ = np.c_[X, Y]

    if measure.measure == 'copula':
        Z = approx_copula(X_) if copula is None else copula
    else:
        Z = X_

    if measure.measure == 'hsic':
        L = measure.label_kernel(Y, Y)
        np.fill_diagonal(L, 0)
        L_ones = L.dot(np.ones(X.shape[0]))
    else:
        L = None
        L_ones = None

    for i in trange(k, leave=False):
        best_score = -np.inf
        best_feature = -1
        if i == 0:
            for j in range(m):
                score = measure.score(Z[:, j], Z[:, -1], L, L_ones)
                if score > best_score:
                    best_score = score
                    best_feature = j
        else:
            for j in (set(np.arange(m)) - set(S)):
                score = 0.0
                for s in S:
                    score += measure.score(Z[:, j], Z[:, s], L, L_ones)
                score = - score / i
                score += measure.score(Z[:, j], Z[:, -1], L, L_ones)
                if score > best_score:
                    best_score = score
                    best_feature = j

        S.append(best_feature)
        subsets.append(copy.deepcopy(S))

    return subsets


def heuristic_selection(X, y, t, measure, estimators, cv=10, regression=True, copula=None):
    """Implement the selection heuristic from Peng and al.

    :X: dataset features
    :y: dataset labels
    :t: desired number of features
    :measure: dependency measure (instance of DependencyMeasure)
    :estimators: list of estimators used to select the best subset (list of tuples (est_name, est))
    :regression: boolean, True if the task is a regression, False if it is a classification
    :copula: copula distribution (optional)
    :returns: dict whose keys are estimators names and values are a tuple (best subset, cv mean, cv std)
    """
    subsets = incremental_search(X, y, t, measure=measure, copula=copula)
    cv_scores = {}

    scoring = 'neg_mean_squared_error' if regression else 'accuracy'
    for est_name, est in estimators:
        scores_tmp = np.zeros((t, 2))
        for i in trange(t, leave=False):
            scores = cross_val_score(est, X[:, subsets[i]], y, cv=cv, scoring=scoring)
            scores_tmp[i, 0] = scores.mean()
            scores_tmp[i, 1] = scores.std()
        best_i = np.argmax(scores_tmp[:, 0])
        cv_scores[est_name] = (subsets[best_i], scores_tmp[best_i, 0], scores_tmp[best_i, 1])

    return cv_scores

