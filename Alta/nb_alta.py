import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.naive_bayes import GaussianNB
import confusionmatrix as cm
import cross_validation as cv
import time
from sklearn.model_selection import GridSearchCV

def NB():
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    clf = GaussianNB()
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    parameters = clf.get_params()
    print(parameters)
    return A_resultado, clf

def ConfusionMatrixNB():
    A_resultado, clf = NB()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    ConfusionMatrixNB = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixNB

def EvaluateNBClouds(ConfusionMatrixNB):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixNB)

def EvaluateNBTotal(ConfusionMatrixNB):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixNB)

def CrossValNB():
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    A_resultado, clf = NB()
    cv.CrossValidation(clf, XaTest[:,0:18], yaTest)

param = {'priors': [None],
        'var_smoothing': [1e-09, 1e-10, 1e-8, 1e-7,1e-5, 1e-1, 1, 10]
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

def NBClass():
    print("-------------------")
    print("Naive Bayes")
    print("Início do CVGrid")
    inicio = time.time()
    A_resultado, clf = NB()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain = XaTrain[:,0:18]
    XaTest = XaTest[:,0:18]
    nb_class = best_model(clf, XaTrain, yaTrain)
    print(nb_class)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return nb_class
