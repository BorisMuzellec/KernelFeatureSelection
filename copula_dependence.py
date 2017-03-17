#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.metrics.pairwise as sk
from sklearn.datasets import load_iris

def approx_copula(X):
      """
      input:
            X: dataset where each line corresponds to a feature vector
      output:
            Z: copula approximation (Z_i,j = rank of x_i wrt the j-th feature)
      """
      return (np.argsort(X, axis = 1)+1)/X.shape[0]
      


def independence_measure(Z, kernel = sk.rbf_kernel):
      """
            Z: copula approximation (Z_i,j = rank of x_i wrt the j-th feature)
      input:
      output:
            kernel copula independence measure of X (where Z = copula(X))
      """
      
      m = Z.shape[0]
      U = np.random.uniform(size = Z.shape)
      
      I = kernel(Z,Z) + kernel(U,U) - kernel(Z,U) - kernel(U,Z)
      np.fill_diagonal(I, 0)
      
      return np.sqrt(I.sum()/(m*(m-1)))
      

def mRMR(X, Y, S, kernel = sk.rbf_kernel):
      """
      Max Relevance - min redundancy kernel copula independence measure
      X: feature dataset
      Y: labels:
      S: index of features to test
      """
      
      n = len(S)
      
      X_ = np.c_[X,Y]
      Z = approx_copula(X_)
      
      MaxRelevance = 0
      minRedundancy = 0
      for i in S:
            MaxRelevance += independence_measure(Z[:, (i, -1)], kernel)
            for j in S:
                  minRedundancy += independence_measure(Z[:, (i, j)], kernel)
      MaxRelevance /= n
      minRedundancy /= n**2
      
      return MaxRelevance - minRedundancy
      pass

iris = load_iris()
X = iris.data
y = iris.target

print mRMR(X,y,[0,1])
print mRMR(X,y,[0,2])
print mRMR(X,y,[0,3])
print mRMR(X,y,[1,2])
print mRMR(X,y,[1,3])
print mRMR(X,y,[2,3])