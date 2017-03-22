#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:30:41 2017

@author: boris
"""
import numpy as np
import copy
import sklearn.metrics.pairwise as sk
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import ensemble

from tqdm import tqdm

from copula_dependence import *

#There's probably room for optimization
def incremental_selection(X, Y, k, method = 'copula', kernel = sk.rbf_kernel):
      """
      Returns all the subsets of features of length smaller than k selected by incremental search
      """
      S = []
      Sets = []
      m = X.shape[1]
      
      X_ = np.c_[X,Y]
      Z = approx_copula(X_)
      
      for i in range(k):
            best_score = -1E3
            best_feature = -1
            if i > 0:
                  for j in list(set(np.arange(m)) - set(S)):
                        score = 0
                        for s in S:
                              score += independence_measure(Z[:, (j,s)], kernel)
                        score = independence_measure(Z[:, (j,-1)], kernel) - score/i
                        if score > best_score:
                              best_score = score
                              best_feature = j
            else:
                  for j in range(m):
                        score = independence_measure(Z[:, (j,-1)], kernel)
                        if score > best_score:
                              best_score = score
                              best_feature = j
                              
            S.append(best_feature)
            Sets.append(copy.deepcopy(S))
                  
      return Sets
      
def selection_heuristic(X, Y, k, classifier, method = 'copula', kernel = sk.rbf_kernel, cv = 5, loss = True):
      """
      The selection heuristic from Peng and al.
      - use incremental selection to find n sequential feature sets (n large)
      - compare the feature sets usign cross validation to find a range k of small error (mean + variance)
      - take the smallest set with smallest error
      """
      
      print("Performing incremental selection")
      S = incremental_selection(X, Y, k, method = method, kernel = kernel)
      cv_scores = {} #Store the 95% confidence interval of the cv_score as [lower_bound, upper_bound]
      
      print("Computing CV scores")
      for i in tqdm(range(k)):
            scores = cross_val_score(classifier, X[:,S[i]], y, cv=5)
            cv_scores[i] = (scores.mean() - 2*scores.std(), scores.mean() + 2*scores.std()) if not loss else (-scores.mean() - 2*scores.std(), -scores.mean() + 2*scores.std())
            
            
      print("Find best score, and undistinguishable scores")     
      #find the highest upper confidence interval bound, then its lower bound, 
      #and all the intervals with upper bound higher than this lower bound
      score_intervals = cv_scores.items()
      #[a for(a,s) in sorted(initial_ranking, key=operator.itemgetter(1), reverse=True)][:10]
      upper_bounds = [u for (s,(l,u)) in score_intervals]
      best_set_idx = np.argmax(upper_bounds)
      best_set_lower_bound =  [l for (s,(l,u)) in score_intervals][best_set_idx]
      
      best_sets = [s for (s,(l,u)) in score_intervals if u >= best_set_lower_bound]
                   
      #Take the smallest best set
      #TODO: implement a clever way of breaking ties
      set_lengths = [len(S[i]) for s in best_sets]
      smallest_best_set = best_sets[np.argmin(set_lengths)]
      
      return S[smallest_best_set], cv_scores[smallest_best_set]
      
boston = load_boston()
X = boston.data
y = boston.target

#print(incremental_selection(X,y,3))

#clf = svm.SVC(kernel='rbf', C=1)

params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.1, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)


print(selection_heuristic(X, y, 6, clf, method = 'copula', kernel = sk.rbf_kernel, cv = 5, loss = True))