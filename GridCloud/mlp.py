import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import time
import evaluatemodel as evm
import seaborn as sns
from sklearn.model_selection import GridSearchCV

## Parâmetros MLP
param = {'activation': ['relu', 'identity', 'logistic', 'tanh'],
        'early_stopping': [False, True],
        'learning_rate': ['constant', 'invscalling', 'adaptive'],
        'learning_rate_init': [0.001, 0.002],
        'max_fun': [12000, 15000, 20000],
        'max_iter': [1500, 2000],
        'momentum': [0.1, 0.5],
        'n_iter_no_change': [5, 10, 15],
        'power_t': [0.5],
        'shuffle': [True, False],
        'solver': ['adam', 'lbfgs', 'sgd'],
        'validation_fraction': [0.05, 0.1],
        'warm_start': [False]
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
def MLPClass():
    print("-------------------")
    print("Pérceptron de múltiplas camadas")
    print("Início do CVGrid")
    inicio = time.time()
    clf = MLPClassifier()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain, XaTest = pre.Chnl(XaTrain, XaTest)
    mlp_class = best_model(clf, XaTrain, yaTrain)
    clf = mlp_class.fit(XaTrain, yaTrain)
    yPred = clf.predict(XaTest)
    final = time.time() - inicio
    min = final/60
    print("Melhores parâmetros: ")
    print(mlp_class)
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return XaTrain, XaTest, yaTrain, yaTest, yPred, clf

def MLPMetrics():
    print("-------------------")
    print("Métricas Percéptron de Multiplas camadas")
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = MLPClass()
    inicio = time.time()
    evm.CrossValidation(clf, XaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")

def PlotMLP():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = MLPClass()
    XaTest = pd.DataFrame(XaTest)
    yPred = pd.DataFrame(yPred)
    df = pd.concat([XaTest, yaTest], axis = 1)
    df.insert(27, 'Predições', yPred.values, allow_duplicates = False)
    sns.relplot(data=df, x="Banda 1", y='Nuvem Alta', hue = 'Predições')
    plt.show()
