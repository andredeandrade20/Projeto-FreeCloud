import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.naive_bayes import GaussianNB
import time
import evaluatemodel as evm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

## Parâmetros Naive Bayes
param = {'priors': [None],
        'var_smoothing': [1e-09, 1e-10, 1e-8, 1e-7,1e-5, 1e-1, 1, 10]
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
def NBClass():
    print("-------------------")
    print("Naive Bayes")
    print("Início do CVGrid")
    inicio = time.time()
    clf = GaussianNB()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    nb_class = best_model(clf, XaTrain, yaTrain)
    clf = nb_class.fit(XaTrain, yaTrain)
    yPred = clf.predict(XaTest)
    final = time.time() - inicio
    min = final/60
    print("Melhores parâmetros: ")
    print(nb_class)
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return XaTrain, XaTest, yaTrain, yaTest, yPred, clf

def NBMetrics():
    print("-------------------")
    print("Métricas Naive Bayes")
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = NBClass()
    inicio = time.time()
    evm.CrossValidation(clf, XaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")

def PlotTree():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = NBClass()
    XaTest = pd.DataFrame(XaTest)
    yPred = pd.DataFrame(yPred)
    df = pd.concat([XaTest, yaTest], axis = 1)
    df.insert(27, 'Predições', yPred.values, allow_duplicates = False)
    sns.relplot(data=df, x="Banda 1", y='Nuvem Alta', hue = 'Predições')
    plt.show()
