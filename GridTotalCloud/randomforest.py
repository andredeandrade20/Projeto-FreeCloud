import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.ensemble import RandomForestClassifier
import time
import evaluatemodel as evm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

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
def RandomForestClass():
    print("-------------------")
    print("RandomForest")
    print("Início do CVGrid")
    inicio = time.time()
    clf = RandomForestClassifier()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    randomforest_class = best_model(clf, XaTrain, yaTrain)
    clf = randomforest_class.fit(XaTrain, yaTrain)
    yPred = clf.predict(XaTest)
    final = time.time() - inicio
    min = final/60
    print("Melhores parâmetros: ")
    print(randomforest_class)
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return XaTrain, XaTest, yaTrain, yaTest, yPred, clf

def RandomForestMetrics():
    print("-------------------")
    print("Métricas Random Forest")
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = RandomForestClass()
    inicio = time.time()
    evm.CrossValidation(clf, XaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")

def PlotRandomForest():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = RandomForestClass()
    XaTest = pd.DataFrame(XaTest)
    yPred = pd.DataFrame(yPred)
    df = pd.concat([XaTest, yaTest], axis = 1)
    df.insert(27, 'Predições', yPred.values, allow_duplicates = False)
    sns.relplot(data=df, x="Banda 1", y='Nuvem Alta', hue = 'Predições')
    plt.show()
