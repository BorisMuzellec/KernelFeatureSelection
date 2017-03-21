#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:30:41 2017

@author: boris
"""
import numpy as np
import sklearn.metrics.pairwise as sk
from sklearn.datasets import load_iris

from copula_dependence import *

#There's probably room for optimization
def incremental_selection(X, Y, k, method = 'copula', kernel = sk.rbf_kernel):
      S = []
      m = X.shape[1]
      
      X_ = np.c_[X,Y]
      Z = approx_copula(X_)
      
      for i in range(k):
            best_score = -np.inf
            best_feature = -1
            if i > 0:
                  for j in list(set(np.arange(m)) - set(S)):
                        score = 0
                        for s in S:
                              score += independence_measure(Z[:, (j,s)], kernel)
                        score = independence_measure(Z[:, (j,-1)], kernel) - score/i
                        if score > best_score:
                              score = best_score
                              best_feature = j
            else:
                  for j in range(m):
                        score = independence_measure(Z[:, (j,-1)], kernel)
                        if score > best_score:
                              score = best_score
                              best_feature = j
                              
            S.append(best_feature)
                  
      return S
      
iris = load_iris()
X = iris.data
y = iris.target

print(incremental_selection(X,y,2))