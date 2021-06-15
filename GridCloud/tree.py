import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.tree import DecisionTreeClassifier
import time
import evaluatemodel as evm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

## Parâmetros Decision Tree
param = {'criterion':['entropy','gini'],
         'splitter': ['best', 'random'],
         'min_samples_leaf':[ 3, 5, 10, 15, 20, 25, 30, 40, 50, 64, 75, 80, 100, 200],
         'max_depth':[5, 6, 8, 10, 12, 15, 18, 20, 25, 27, 30, 35, 40, 45, 50, 56, 60, 64, 100, 150, 200],
         'max_features': [None, 'auto', 'sqrt', 'log2'],
         'min_samples_split':[0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.6, 0.75, 0.8, 0.9, 1.0],
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

## Aplicação das funções de gridsearch para os melhores parâmetros da árvore de decisão
def TreeClass():
    print("-------------------")
    print("Decision Tree")
    print("Início do CVGrid")
    inicio = time.time()
    clf = DecisionTreeClassifier()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain, XaTest = pre.Chnl(XaTrain, XaTest)
    tree_class = best_model(clf, XaTrain, yaTrain)
    clf = tree_class.fit(XaTrain, yaTrain)
    yPred = clf.predict(XaTest)
    final = time.time() - inicio
    min = final/60
    print("Melhores parâmetros: ")
    print(tree_class)
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return XaTrain, XaTest, yaTrain, yaTest, yPred, clf


def TreeMetrics():
    print("-------------------")
    print("Métricas Árvore de decisão")
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = TreeClass()
    inicio = time.time()
    evm.CrossValidation(clf, XaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")

def PlotTree():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = TreeClass()
    XaTest = pd.DataFrame(XaTest)
    yPred = pd.DataFrame(yPred)
    df = pd.concat([XaTest, yaTest], axis = 1)
    df.insert(27, 'Predições', yPred.values, allow_duplicates = False)
    df.insert(28, 'Banda', df.iloc[:,0], allow_duplicates = False)
    sns.relplot(data=df, x="Banda", y='Nuvem Alta', hue = 'Predições')
    plt.show()
