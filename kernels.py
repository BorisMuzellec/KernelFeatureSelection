#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:36:32 2017

@author: boris
"""
from collections import Counter

import numpy as np

# Label kernels


def binary_clf_kernel(y, normalize=False):
    n = y.shape[0]
    L = -1 * np.ones((n, n))

    for i in range(n):
        for j in range(n):
            if y[i] == y[j]:
                L[i, j] = 1

    return L


def multiclass_clf_kernel(y, normalize=False):
    n = y.shape[0]
    counts = Counter(y)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            L[i, j] = 1. / counts[y[i]] if y[i] == y[j] else 0

    return L
