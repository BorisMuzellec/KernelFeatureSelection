#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:13:18 2017

@author: boris
"""

#Copula-free kernel dependency
import sklearn.metrics.pairwise as sk
import numpy as np


#The unbiased estimator
def kernel_dependency(X,Y, feature_kernel = sk.rbf_kernel, label_kernel = sk.rbf_kernel):
      """
      Allows to use different kernels for features and labels, which is usefull in the case of classification
      For regression, linear or rbf kernels work well
      """
      m = X.shape[0]
      
      K = feature_kernel(X,X)
      L = label_kernel(Y,Y)
      
      np.fill_diagonal(K,0)
      np.fill_diagonal(L,0)
      
      KL = K.dot(L)
      return (np.trace(KL) + K.sum()*L.sum()/((m-1)*(m-2)) - 2./(m-2)*KL.sum())/(m*(m-3))
      