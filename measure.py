#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:34:16 2017

@author: boris
"""

import numpy as np
import sklearn.metrics.pairwise as sk
from copula_dependence import copula_measure
from kernel_dependency import hsic_approx

class Dependency_Measure(object):
    """Rank recipients from the adress books of an user based on k-NN classification
    on the tf-idf representations of the emails."""

    def __init__(self, measure = 'copula', feature_kernel=sk.rbf_kernel, label_kernel=sk.rbf_kernel, gamma = 1./12):
        """
        Args:
            emails_info (pandas.DataFrame): dataframe with emails info (mid, date, body, recipients, sender)
        """
        
        assert(measure in ['copula', 'hsic', 'mutual_information'])
        
        self.measure= measure
        self.feature_kernel = feature_kernel
        self.label_kernel = label_kernel
        self.gamma = gamma
        
        if self.measure == 'copula':
              self.scorer = lambda x,y : copula_measure(x,y, kernel=self.feature_kernel, gamma= self.gamma)
        elif self.measure == 'hsic':
              self.scorer = lambda x, y: hsic_approx(x, y, feature_kernel = self.feature_kernel, label_kernel = self.label_kernel, gamma = self.gamma)
              

        
    def score(self, X, Y):
          """
          Return the dependency score I(X,Y)
          """
          X_ = X[:,np.newaxis] if len(X.shape) == 1 else X
          Y_ = Y[:,np.newaxis] if len(Y.shape) == 1 else Y
          return self.scorer(X_,Y_)