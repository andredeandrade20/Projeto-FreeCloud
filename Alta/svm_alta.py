import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.svm import LinearSVC
import confusionmatrix as cm
import time
from sklearn.model_selection import GridSearchCV

def SVC():
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    clf = LinearSVC(max_iter = 12000)
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixSVC():
    A_resultado, clf = SVC()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    ConfusionMatrixSVC = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixSVC

def EvaluateSVCClouds(ConfusionMatrixSVC):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixSVC)

def EvaluateSVCTotal(ConfusionMatrixSVC):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixSVC)

param = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'class_weight': [None, 'balanced'],
        'fit_intercept': [False, True],
        'intercept_scaling': [0, 0.5, 1],
        'loss': ['hinge', 'squared_hinge'],
        'max_iter': [10000,11500, 12000, 13500, 15000],
        'multi_class': ['ovr', 'crammer_singer'],
        'penalty': ['l1','l2'],
        'tol': [0.0001, 0.0005, 0.001, 0.005],
        'verbose': [1, 2, 4, 5, 10, 20]
        }

def config_param(clf, param, cv = 5, n_jobs = 1, scoring = 'balanced_accuracy'):
    grid_class = GridSearchCV(clf, param, cv = cv, n_jobs = n_jobs, scoring = scoring)
    return clf, grid_class

def get_param(clf, param, X, y):
    clf, grid_class = config_param(clf, param)
    return grid_class.fit(X,y)

def best_model(clf, X, Y):
    all_param = get_param(clf, param, X, Y)
    best_result = all_param.best_estimator_
    return best_result

def SVMClass():
    print("-------------------")
    print("Support Vector Machine")
    print("Início do CVGrid")
    inicio = time.time()
    A_resultado, clf = SVC()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    svm_class = best_model(clf, XaTrain, yaTrain)
    print(svm_class)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return svm_class
