#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:00:36 2017

@author: boris
"""
import os
import pickle

import numpy as np
from sklearn import ensemble, linear_model, svm
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes
import sklearn.metrics.pairwise as sk

from copula_dependence import approx_copula
from feature_selection_algorithms import *
from measure import Dependency_Measure


BASE_DIR = os.path.abspath(os.path.curdir)


boston = load_boston()
breast_cancer = load_breast_cancer()
diabetes = load_diabetes()


breast_copula = approx_copula(np.c_[breast_cancer.data, breast_cancer.target])

# HSIC = Dependency_Measure(measure='hsic', feature_kernel=sk.rbf_kernel,
                          # label_kernel=sk.rbf_kernel, gamma=1. / 12)
# COPULA = Dependency_Measure(
    # measure='copula', feature_kernel=sk.rbf_kernel, gamma=6)
# MUTUAL_INFO = Dependency_Measure(
    # measure='mutual_information', feature_kernel=sk.rbf_kernel, gamma=6)

print('=== Classification Problems ===')


boost_params = {'loss': 'deviance', 'learning_rate': 0.1, 'n_estimators': 100,
                'subsample': 1.0, 'max_depth': 3}
svm_params = {'C': 1.0, 'kernel': 'rbf', 'degree': 3, 'gamma': 'auto'}

BoostingClassifier = ensemble.GradientBoostingClassifier(**boost_params)
SVMClassifier = svm.SVC(**svm_params)

classifiers = [SVMClassifier, BoostingClassifier]

print('Classification Problem: breast_cancer Dataset')


accuracy_scores = {}
measures = ['mutual_information', 'hsic', 'copula']
kernels = [sk.rbf_kernel, sk.polynomial_kernel]

# Compare seletion methods for hsic and copula on rbf kernel
# with a svm classifier and a gradient boosting classifiers
print("\tHSIC vs Copula (rbf kernel, SVM + Gradient Boosting classifiers)")
alg_accuracy_scores = {}
for measure in ['hsic', 'copula']:
    alg_accuracy_scores[measure] = {}
    alg_accuracy_scores[measure]['backward'] = {}
    alg_accuracy_scores[measure]['forward'] = {}
    alg_accuracy_scores[measure]['heuristic'] = {}

    for clf in classifiers:
        alg_accuracy_scores[measure]['backward'][clf] = backward_selection(
                breast_cancer.data, breast_cancer.target, 6,
                classifier=clf, measure=Dependency_Measure(measure=measure),
                regression=False, copula=breast_copula)
        alg_accuracy_scores[measure]['forward'][clf] = forward_selection(
                breast_cancer.data, breast_cancer.target, 6,
                classifier=clf, measure=Dependency_Measure(measure=measure),
                regression=False, copula=breast_copula)
        alg_accuracy_scores[measure]['heuristic'][clf] = selection_heuristic(
                breast_cancer.data, breast_cancer.target, 6,
                classifier=clf, measure=Dependency_Measure(measure=measure),
                regression=False, copula=breast_copula)


# Compare hsic and copula on rbf and polynomial kernels
# using heuristic selection with a linear regressor and a gradient
# boosting regressor
print("\tHSIC vs Copula (polynomial kernel, Linear Model + Gradient Boosting classifiers)")
for measure in ['hsic', 'copula']:
    accuracy_scores[measure] = {}
    for kernel in kernels:
        accuracy_scores[measure][kernel] = {}
        for clf in classifiers:
            accuracy_scores[measure][kernel][clf] = selection_heuristic(
                    breast_cancer.data, breast_cancer.target, 6,
                    clf, measure=Dependency_Measure(measure=measure, feature_kernel=kernel),
                    regression=False, copula=breast_copula)

# Compare mutual_information and hsic and copula with linear kernels
# using heuristic selection with a svm classifier and a gradient boosting
# classifier
print("\tMutual Information vs HSIC vs Copula (linear kernels, heuristic selection, SVM + Gradient Boosting classifiers)")
Linear_accuracy_scores = {}
for measure in ['hsic', 'copula']:
    Linear_accuracy_scores[measure] = {}
    for clf in classifiers:
        Linear_accuracy_scores[measure][clf] = selection_heuristic(breast_cancer.data, breast_cancer.target, 6,
                                                                   clf, measure=Dependency_Measure(measure=measure, feature_kernel=sk.linear_kernel), regression=False, copula=breast_copula)

print('Saving Classification Benchmark')

breast_cancer_benchmark = [accuracy_scores,
                           Linear_accuracy_scores, alg_accuracy_scores]

with open('breast_cancer_Benchmark.dat', "wb") as f:
    pickle.dump(len(breast_cancer_benchmark), f)
    for value in breast_cancer_benchmark:
        pickle.dump(value, f)

# params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
          # 'learning_rate': 0.1, 'loss': 'ls'}

# BoostingRegressor = ensemble.GradientBoostingRegressor(**params)
# LinearRegressor = linear_model.LinearRegression()

# regressors = [LinearRegressor, BoostingRegressor]

# print('Regression Problem: Boston Housing Dataset')


# L2_scores = {}
# measures = ['mutual_information', 'hsic', 'copula']
# kernels = [sk.rbf_kernel, sk.polynomial_kernel]

# # Compare seletion methods for hsic and copula on rbf kernel
# # with a linear regressor and a gradient boosting regressor
# alg_L2_scores = {}
# for measure in ['hsic', 'copula']:
    # alg_L2_scores[measure] = {}
    # alg_L2_scores[measure]['backward'] = {}
    # alg_L2_scores[measure]['forward'] = {}
    # alg_L2_scores[measure]['heuristic'] = {}

    # for regressor in regressors:
        # alg_L2_scores[measure]['backward'][regressor] = backward_selection(
                # boston.data, boston.target, 6,
                # classifier=regressor, measure=Dependency_Measure(measure=measure))
        # alg_L2_scores[measure]['forward'][regressor] = forward_selection(
                # boston.data, boston.target, 6,
                # classifier=regressor, measure=Dependency_Measure(measure=measure))
        # alg_L2_scores[measure]['heuristic'][regressor] = selection_heuristic(
                # boston.data, boston.target, 6,
                # regressor, measure=Dependency_Measure(measure=measure))


# # Compare hsic and copula on rbf and polynomial kernels
# # using heuristic selection with a linear regressor and a gradient
# # boosting regressor
# for measure in ['hsic', 'copula']:
    # L2_scores[measure] = {}
    # for kernel in kernels:
        # L2_scores[measure][kernel] = {}
        # for regressor in regressors:
            # L2_scores[measure][kernel][regressor] = selection_heuristic(
                    # boston.data, boston.target, 6,
                    # regressor, measure=Dependency_Measure(measure=measure, feature_kernel=kernel))

# # Compare mutual_information and hsic and copula with linear kernels
# # using heuristic selection with a linear regressor and a gradient
# # boosting regressor

# Linear_L2_scores = {}
# for measure in ['hsic', 'copula']:
    # Linear_L2_scores[measure] = {}
    # for regressor in regressors:
        # Linear_L2_scores[measure][regressor] = selection_heuristic(
                # boston.data, boston.target, 6,
                # regressor, measure=Dependency_Measure(measure=measure, feature_kernel=sk.linear_kernel))


# # TODO: Kernel parameter comparison

# print('Saving Regression Benchmark')

# boston_benchmark = [L2_scores, Linear_L2_scores, alg_L2_scores]

# with open('Boston_Benchmark.dat', "wb") as f:
    # pickle.dump(len(boston_benchmark), f)
    # for value in boston_benchmark:
        # pickle.dump(value, f)

# # with open(os.path.join(BASE_DIR, 'Boston_Benchmark.npy'), 'wb') as f:
# #      np.save(f, L2_scores)
