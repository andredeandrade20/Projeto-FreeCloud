import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.linear_model import LogisticRegression
import confusionmatrix as cm
import time
from sklearn.model_selection import GridSearchCV

def LR():
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain = XaTrain[:,0:18]
    XaTest = XaTest[:,0:18]
    clf = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter = 200)
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    p = clf.get_params()
    print(p)
    return A_resultado, clf

def ConfusionMatrixLR():
    A_resultado, clf = LR()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    ConfusionMatrixLR = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixLR

def EvaluateLRClouds(ConfusionMatrixLR):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixLR)

def EvaluateLRTotal(ConfusionMatrixLR):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixLR)

param = {'C': [0.1, 0.25, 0.5, 1.0],
        'class_weight': [None, 'balanced'],
        'dual': [False, True],
        'fit_intercept': [True, False],
        'max_iter': [200, 400, 800],
        'multi_class': ['multinomial'],
        'n_jobs': [None],
        'penalty': ['l2', 'l1', 'elasticnet', 'None'],
        'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
        'verbose': [0,1],
        'warm_start': [False, True]
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

def LRClass():
    print("-------------------")
    print("Regressão Logística")
    print("Início do CVGrid")
    inicio = time.time()
    A_resultado, clf = LR()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    lr_class = best_model(clf, XaTrain, yaTrain)
    print(lr_class)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return lr_class
