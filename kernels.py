#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:36:32 2017

@author: boris
"""

import numpy as np

#Label kernels
def binary_clf_kernel(y, normalize = False):
      
      n = len(y)
      L = np.zeros((y, y))
      
      for i in range(n):
            for j in range(n):
                  L[i,j] = 1 if y[i] == y[j] else -1
      
      return L
      
def multiclass_clf_kernel(y, normalize = False):
      
      n = len(y)
      
      labels = y.unique()      
      counts = {}
      for l in labels:
            counts[l] = y.count(l)
            
      L = np.zeros((n,n))
      
      for i in range(n):
            for j in range(n):
                  L[i,j] = 1./counts[y[i]] if y[i] == y[j] else 0
                  
      return L
      
      