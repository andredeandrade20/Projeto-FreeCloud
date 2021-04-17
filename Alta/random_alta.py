import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.ensemble import RandomForestClassifier
import confusionmatrix as cm
import time
from sklearn.model_selection import GridSearchCV

def RandomTree():
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixRandomTree():
    A_resultado, clf = RandomTree()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    ConfusionMatrixRandomTree = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixRandomTree

def EvaluateRandomTreeClouds(ConfusionMatrixRandomTree):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixRandomTree)

def EvaluateRandomTreeTotal(ConfusionMatrixRandomTree):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixRandomTree)

param = {'bootstrap': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        'criterion': ['entropy', 'gini'],
        'max_depth':[5, 6, 8, 10, 12, 15, 18, 20, 25, 27, 30, 35, 40, 45, 50, 56, 60, 64, 100, 150, 200],
        'max_features': ['auto', 'log2', 'sqrt'],
        'max_samples': [1, 2, 5, 10, 20],
        'min_samples_leaf': [3, 5, 10, 15, 20, 30, 50, 100, 200],
        'min_samples_split': [0.1, 0.25, 0.5, 0.75, 1.0],
        'n_estimators': [10, 20, 50],
        'oob_score': [True, False],
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

def RandomTreeClass():
    print("-------------------")
    print("Random Forest")
    print("Início do CVGrid")
    inicio = time.time()
    A_resultado, clf = RandomTree()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain = XaTrain[:,0:18]
    XaTest = XaTest[:,0:18]
    random_class = best_model(clf, XaTrain, yaTrain)
    print(random_class)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return random_class
