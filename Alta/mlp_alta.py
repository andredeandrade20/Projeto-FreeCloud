import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.neural_network import MLPClassifier
import confusionmatrix as cm
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV

def MLP():
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    clf = MLPClassifier()
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixMLP():
    A_resultado, clf = MLP()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    ConfusionMatrixMLP = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixMLP

def EvaluateMLPClouds(ConfusionMatrixMLP):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixMLP)

def EvaluateMLPTotal(ConfusionMatrixMLP):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixMLP)

param = {'activation': ['relu', 'identity', 'logistic', 'tanh'],
        'early_stopping': [False, True],
        'learning_rate': ['constant', 'invscalling', 'adaptive'],
        'learning_rate_init': [0.001, 0.002],
        'max_fun': [10000, 12000, 15000],
        'max_iter': [1500, 2000],
        'momentum': [0.1, 0.5],
        'n_iter_no_change': [5, 10, 15],
        'power_t': [0.5],
        'shuffle': [True, False],
        'solver': ['adam', 'lbfgs', 'sgd'],
        'validation_fraction': [0.05, 0.1],
        'warm_start': [False]
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

def MLPClass():
    print("-------------------")
    print("Perceptron de múltiplas camadas")
    print("Início do CVGrid")
    inicio = time.time()
    A_resultado, clf = MLP()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    mlp_class = best_model(clf, XaTrain, yaTrain)
    print(mlp_class)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return mlp_class
