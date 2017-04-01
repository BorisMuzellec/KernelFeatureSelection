import numpy as np
from sklearn import ensemble, linear_model, svm
import sklearn.metrics.pairwise as sk

from copula_dependency import approx_copula
import feature_selection_algorithms as algos
from measure import DependencyMeasure


def compare_algorithms(X, y, estimators, measure_name, copula=None):
    alg_accuracy_scores = {}
    measure = DependencyMeasure(measure=measure_name)

    print("Measure '{}', algorithm: 'backward_selection'".format(measure_name))
    alg_accuracy_scores['backward'] = algos.backward_selection(X, y, 6, measure, 
            estimators, cv=10, regression=False, copula=copula)
    print("Measure '{}', algorithm: 'forward_selection'".format(measure_name))
    alg_accuracy_scores['forward'] = algos.forward_selection(X, y, 6, measure, 
            estimators, cv=10, regression=False, copula=copula)
    print("Measure '{}', algorithm: 'heuristic_selection'".format(measure_name))
    alg_accuracy_scores['heuristic'] = algos.heuristic_selection(X, y, 6, measure, 
            estimators, cv=10, regression=False, copula=copula)
    
    return alg_accuracy_scores


def compare_kernels(X, y, estimators, measure_name, copula=None):
    kern_accuracy_scores = {}

    print("Measure '{}', kernel: 'rbf'".format(measure_name))
    measure = DependencyMeasure(measure=measure_name, feature_kernel=sk.rbf_kernel)
    kern_accuracy_scores['rbf'] = algos.heuristic_selection(X, y, 6, measure, 
            estimators, regression=False, copula=copula)
    print("Measure '{}', kernel: 'polynomial'".format(measure_name))
    measure = DependencyMeasure(measure=measure_name, feature_kernel=sk.polynomial_kernel)
    kern_accuracy_scores['polynomial'] = algos.heuristic_selection(X, y, 6, measure, 
            estimators, regression=False, copula=copula)
    print("Measure '{}', kernel: 'linear'".format(measure_name))
    measure = DependencyMeasure(measure=measure_name, feature_kernel=sk.linear_kernel)
    kern_accuracy_scores['linear'] = algos.heuristic_selection(X, y, 6, measure, 
            estimators, regresstion=False, copula=copula)

    return kern_accuracy_scores

