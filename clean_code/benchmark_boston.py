import pickle

import numpy as np
from sklearn import ensemble, linear_model, svm
from sklearn.datasets import load_boston
import sklearn.metrics.pairwise as sk

import benchmark_tools as bm
from copula_dependency import approx_copula
from feature_selection_algorithms import heuristic_selection
from measure import DependencyMeasure


boston = load_boston()
X, y = boston.data, boston.target
boston_copula = approx_copula(np.c_[X, y])

gb_params = {'loss': 'ls', 'learning_rate': 0.1, 
             'n_estimators': 100, 'subsample': 1.0, 'max_depth': 3}
svm_params = {'C': 1.0, 'kernel': 'rbf', 'degree': 3, 'gamma': 'auto'}

results_algs = {}
for measure_name in ['hsic', 'copula']:
    gb_clf = ensemble.GradientBoostingRegressor(**gb_params)
    svm_clf = svm.SVR(**svm_params)
    classifiers = [('gb', gb_clf), ('svm', svm_clf)]
    results_algs[measure_name] = bm.compare_algorithms(X, y, estimators=classifiers, 
            measure_name=measure_name, copula=boston_copula, regression=True)

results_kerns = {}
for measure_name in ['hsic', 'copula', 'mutual_information']:
    gb_clf = ensemble.GradientBoostingRegressor(**gb_params)
    svm_clf = svm.SVR(**svm_params)
    classifiers = [('gb', gb_clf), ('svm', svm_clf)]
    results_kerns[measure_name] = bm.compare_kernels(X, y, estimators=classifiers, 
            measure_name=measure_name, copula=boston_copula, regression=True)

results_linear = {}
for measure_name in ['hsic', 'copula', 'mutual_information']:
    gb_clf = ensemble.GradientBoostingRegressor(**gb_params)
    svm_clf = svm.SVR(**svm_params)
    classifiers = [('gb', gb_clf), ('svm', svm_clf)]
    measure = DependencyMeasure(measure=measure_name, feature_kernel=sk.linear_kernel)
    results_linear[measure_name] = heuristic_selection(X, y, 6, measure, classifiers, cv=10, regression=True, copula=boston_copula)


results = [('results_algs', results_algs), ('results_kerns', results_kerns), ('results_linear', results_linear)]


with open('boston_benchmark', 'wb') as f:
    pickle.dump(len(results), f)
    for r in results:
        pickle.dump(r, f)

