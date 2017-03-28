#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:00:36 2017

@author: boris
"""

import numpy as np
import sklearn.metrics.pairwise as sk
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes
from sklearn import svm
from sklearn import ensemble

from feature_selection_algorithms import *
from measure import Dependency_Measure
# There's probably room for optimization



boston = load_boston()
breast_cancer = load_breast_cancer()
diabetes = load_diabetes


HSIC = Dependency_Measure(measure='hsic', feature_kernel=sk.rbf_kernel,
                          label_kernel=sk.rbf_kernel, gamma=1. / 12)

COPULA = Dependency_Measure(
    measure='copula', feature_kernel=sk.rbf_kernel, gamma=6)
 
MUTUAL_INFO = Dependency_Measure(measure='mutual_information', feature_kernel=sk.rbf_kernel, gamma=6)

params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.1, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)


print('Regression Problem: Boston Housing Dataset')


L2_scores = {}
measures = ['mutual_information', 'hsic', 'copula']
kernels = [sk.rbf_kernel, sk.polynomial_kernel]


for measure in ['hsic','copula']:
      L2_scores[measure] = {}
      for kernel in kernels:
            L2_scores[measure][kernel] =  selection_heuristic(boston.data, boston.target, 10,
            clf, measure = Dependency_Measure(measure=measure, feature_kernel = kernel))


print L2_scores

print('Classification Problems')
