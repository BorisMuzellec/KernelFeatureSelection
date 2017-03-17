#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.metrics.pairwise as sk


def approx_copula(X):
      """
      input:
            X: dataset where each line corresponds to a feature vector
      output:
            Z: copula approximation (Z_i,j = rank of x_i wrt the j-th feature)
      """
      return (np.argsort(X, axis = 1)+1)/X.shape[0]
      

# It might be better to compute the copula only once ?
def independence_measure(X, kernel):
      
      m = X.shape[0]
      
      U = np.random.uniform(size = m)
      Z = approx_copula(X)
      
      return np.sqrt(np.fill_diagonal(kernel(Z,Z) + kernel(U,U) - kernel(Z,U) - kernel(U,Z), 0).sum()/(m*(m-1)))

