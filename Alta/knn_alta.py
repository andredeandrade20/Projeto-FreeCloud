import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.neighbors import KNeighborsClassifier
import confusionmatrix as cm
import time
from sklearn.model_selection import GridSearchCV

def KNN():
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain = XaTrain[:,0:18]
    XaTest = XaTest[:,0:18]
    clf = KNeighborsClassifier(n_neighbors = 5, p = 2)
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixKNN():
    A_resultado, clf = KNN()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    ConfusionMatrixKNN = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixKNN

def EvaluateKNNClouds(ConfusionMatrixKNN):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixKNN)

def EvaluateKNNTotal(ConfusionMatrixKNN):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixKNN)

param = {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [30, 32, 35, 40,  50, 100],
        'metric': ['minkowski'],
        'n_neighbors': [2, 3, 4, 5, 7, 10, 12, 20, 50,],
        'p': [1, 2, 5, 10],
        'weights': ['uniform', 'distance']
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

def KNNClass():
    print("-------------------")
    print("K-Neirest Neighbors")
    print("Início do CVGrid")
    inicio = time.time()
    A_resultado, clf = KNN()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    knn_class = best_model(clf, XaTrain, yaTrain)
    print(knn_class)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return knn_class
