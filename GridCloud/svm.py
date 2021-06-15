import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.svm import LinearSVC
import time
import evaluatemodel as evm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

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
def SVMClass():
    print("-------------------")
    print("Support Vector Machine")
    print("Início do CVGrid")
    inicio = time.time()
    clf = LinearSVC()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain, XaTest = pre.Chnl(XaTrain, XaTest)
    svm_class = best_model(clf, XaTrain, yaTrain)
    clf = svm_class.fit(XaTrain, yaTrain)
    yPred = clf.predict(XaTest)
    final = time.time() - inicio
    min = final/60
    print("Melhores parâmetros: ")
    print(svm_class)
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return XaTrain, XaTest, yaTrain, yaTest, yPred, clf

def SVMMetrics():
    print("-------------------")
    print("Métricas Support Vector Machine")
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = SVMClass()
    inicio = time.time()
    evm.CrossValidation(clf, XaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")

def PlotSVM():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = SVMClass()
    XaTest = pd.DataFrame(XaTest)
    yPred = pd.DataFrame(yPred)
    df = pd.concat([XaTest, yaTest], axis = 1)
    df.insert(27, 'Predições', yPred.values, allow_duplicates = False)
    sns.relplot(data=df, x="Banda 1", y='Nuvem Alta', hue = 'Predições')
    plt.show()
