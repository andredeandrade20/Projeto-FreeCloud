import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.linear_model import LogisticRegression
import time
import evaluatemodel as evm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

## Parâmetros Regressão Logística
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
def LRClass():
    print("-------------------")
    print("Pérceptron de múltiplas camadas")
    print("Início do CVGrid")
    inicio = time.time()
    clf = LogisticRegression()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    lr_class = best_model(clf, XaTrain, yaTrain)
    clf = lr_class.fit(XaTrain, yaTrain)
    yPred = clf.predict(XaTest)
    final = time.time() - inicio
    min = final/60
    print("Melhores parâmetros: ")
    print(lr_class)
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return XaTrain, XaTest, yaTrain, yaTest, yPred, clf

def LRMetrics():
    print("-------------------")
    print("Métricas Regressão Logística")
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = LRClass()
    inicio = time.time()
    evm.CrossValidation(clf, XaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")

def PlotLR():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = LRClass()
    XaTest = pd.DataFrame(XaTest)
    yPred = pd.DataFrame(yPred)
    df = pd.concat([XaTest, yaTest], axis = 1)
    df.insert(27, 'Predições', yPred.values, allow_duplicates = False)
    sns.relplot(data=df, x="Banda 1", y='Nuvem Alta', hue = 'Predições')
    plt.show()
