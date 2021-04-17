import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.tree import DecisionTreeClassifier
import confusionmatrix as cm
from sklearn.model_selection import GridSearchCV
import time
from sklearn import metrics
import cross_validation as cv

def Tree():
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain, XaTest = pre.Chnl(XaTrain,XaTest)
    clf = DecisionTreeClassifier(max_depth=12, min_samples_leaf=3, min_samples_split=0.1)
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf, XaTrain, XaTest, yaTrain, yaTest

def CrossValTree():
    A_resultado, clf, XaTrain, XaTest, yaTrain, yaTest = Tree()
    cv.CrossValidation(clf, XaTrain, yaTrain)

def ConfusionMatrixTree():
    A_resultado, clf = Tree()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain = XaTrain[:,0:18]
    XaTest = XaTest[:,0:18]
    ConfusionMatrixTree = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixTree

def EvaluateTreeClouds(ConfusionMatrixTree):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixTree)

def EvaluateTreeTotal(ConfusionMatrixTree):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixTree)

param = {'criterion':['entropy','gini'],
         'splitter': ['best', 'random'],
         'min_samples_leaf':[ 3, 5, 10, 15, 20, 25, 30, 40, 50, 64, 75, 80, 100, 200],
         'max_depth':[5, 6, 8, 10, 12, 15, 18, 20, 25, 27, 30, 35, 40, 45, 50, 56, 60, 64, 100, 150, 200],
         'max_features': [None, 'auto', 'sqrt', 'log2'],
         'min_samples_split':[0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.6, 0.75, 0.8, 0.9, 1.0],
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

def TreeClass():
    print("-------------------")
    print("Decision Tree")
    print("Início do CVGrid")
    inicio = time.time()
    A_resultado, clf = Tree()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain = XaTrain[:,0:18]
    XaTest = XaTest[:,0:18]
    tree_class = best_model(clf, XaTrain, yaTrain)
    print(tree_class)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final do CVGrid')
    print("-------------------")
    return tree_class

def multi_metrics(y_true, y_pred):
    accur = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average = "weighted")
    recall = metrics.recall_score(y_true, y_pred, average = "weighted")
    f1 = metrics.f1_score(y_true, y_pred,average = "weighted")
    fbeta = metrics.fbeta_score(y_true, y_pred, beta=0.5,average = "weighted")
    balanced_score = metrics.balanced_accuracy_score(y_true, y_pred)
    MMC = metrics.matthews_corrcoef(y_true, y_pred)
    print(accur, precision, recall, f1, fbeta, balanced_score, MMC)
    return accur, precision, recall, f1, fbeta, balanced_score, MMC

def TreeMetrics():
    print("-------------------")
    print("Métricas Árvore de decisão")
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    A_resultado, clf = Tree()
    tree_metrics = multi_metrics(XaTrain, yaTrain)
    print("-------------------")
