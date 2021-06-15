import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.neighbors import KNeighborsClassifier
import time
import evaluatemodel as evm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

## Parâmetros K-Neirest Neighbors
param = {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [30, 32, 35, 40,  50, 100],
        'metric': ['minkowski'],
        'n_neighbors': [2, 3, 4, 5, 7, 10, 12, 20, 50,],
        'p': [1, 2, 5, 10],
        'weights': ['uniform', 'distance']
        }


##Funções de GridSearchCV
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

## Aplicação das funções de gridsearch para os melhores parâmetros da máquina de vetor suporte
def KNNClass():
    print("-------------------")
    print("K-Neirest Neighbors")
    print("Início do CVGrid")
    inicio = time.time()
    clf = KNeighborsClassifier()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain, XaTest = pre.Chnl(XaTrain, XaTest)
    knn_class = best_model(clf, XaTrain, yaTrain)
    clf = knn_class.fit(XaTrain, yaTrain)
    yPred = clf.predict(XaTest)
    final = time.time() - inicio
    min = final/60
    print("Melhores parâmetros: ")
    print(knn_class)
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return XaTrain, XaTest, yaTrain, yaTest, yPred, clf

def KNNMetrics():
    print("-------------------")
    print("Métricas K-Neirest Neighbors")
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = KNNClass()
    inicio = time.time()
    evm.CrossValidation(clf, XaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")

def PlotKNN():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = KNNClass()
    XaTest = pd.DataFrame(XaTest)
    yPred = pd.DataFrame(yPred)
    df = pd.concat([XaTest, yaTest], axis = 1)
    df.insert(27, 'Predições', yPred.values, allow_duplicates = False)
    sns.relplot(data=df, x="Banda 1", y='Nuvem Alta', hue = 'Predições')
    plt.show()
