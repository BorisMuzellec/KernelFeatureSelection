import pickle

import numpy as np
from sklearn import ensemble, linear_model, svm
from sklearn.datasets import load_breast_cancer
import sklearn.metrics.pairwise as sk

import benchmark_tools as bm
from copula_dependency import approx_copula
from feature_selection_algorithms import heuristic_selection
from measure import DependencyMeasure


breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
breast_copula = approx_copula(np.c_[X, y])

gb_params = {'loss': 'deviance', 'learning_rate': 0.1, 
             'n_estimators': 100, 'subsample': 1.0, 'max_depth': 3}
svm_params = {'C': 1.0, 'kernel': 'rbf', 'degree': 3, 'gamma': 'auto'}

results_algs = {}
for measure_name in ['hsic', 'copula']:
    gb_clf = ensemble.GradientBoostingClassifier(**gb_params)
    svm_clf = svm.SVC(**svm_params)
    classifiers = [('gb', gb_clf), ('svm', svm_clf)]
    results_algs[measure_name] = bm.compare_algorithms(X, y, estimators=classifiers, 
            measure_name=measure_name, copula=breast_copula)

results_kerns = {}
for measure_name in ['hsic', 'copula', 'mutual_information']:
    gb_clf = ensemble.GradientBoostingClassifier(**gb_params)
    svm_clf = svm.SVC(**svm_params)
    classifiers = [('gb', gb_clf), ('svm', svm_clf)]
    results_kerns[measure_name] = bm.compare_kernels(X, y, estimators=classifiers, 
            measure_name=measure_name, copula=breast_copula)

results_linear = {}
for measure_name in ['hsic', 'copula', 'mutual_information']:
    gb_clf = ensemble.GradientBoostingClassifier(**gb_params)
    svm_clf = svm.SVC(**svm_params)
    classifiers = [('gb', gb_clf), ('svm', svm_clf)]
    measure = DependencyMeasure(measure=measure_name, feature_kernel=sk.linear_kernel)
    results_linear[measure_name] = heuristic_selection(X, y, 6, measure, classifiers, cv=10, regression=False, copula=breast_copula)


results = [('results_algs', results_algs), ('results_kerns', results_kerns), ('results_linear', results_linear)]


with open('breast_cancer_benchmark', 'wb') as f:
    pickle.dump(len(results), f)
    for r in results:
        pickle.dump(r, f)

