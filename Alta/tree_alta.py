import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.tree import DecisionTreeClassifier
import confusionmatrix as cm
from sklearn.model_selection import GridSearchCV

def Tree():
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain = XaTrain[:,0:18]
    XaTest = XaTest[:,0:18]
    clf = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best')
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

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

param = {'criterion':['gini','entropy'],
         'splitter': ['best', 'random'],
         'min_samples_leaf':[ 3, 5, 10, 15, 20, 25, 30, 40, 50, 100, 200],
         'max_depth':[5, 6, 8, 10, 12, 15, 18, 20, 25, 27, 30, 35, 40, 45, 50, 56, 60, 64, 100, 150, 200],
         'max_features': [None, 'auto', 'sqrt', 'log2'],
         'min_samples_split':[0.2, 0.3, 0.5, 0.6, 0.75, 0.8, 0.9, 1.0]
         }

def config_param(clf, param, cv = None, n_jobs = 1, scoring = 'balanced_accuracy'):
    grid_class = GridSearchCV(clf, param, cv = cv, n_jobs = n_jobs, scoring = scoring)
    return clf, grid_class

def get_param(clf, param, X, y):
    clf, grid_class = config_param(clf, param)
    return grid_class.fit(X,y)

def best_model(clf, X, Y):
    print("-------------------")
    print("Início do CVGrid")
    all_param = get_param(clf, param, X, Y)
    best_result = all_param.best_estimator_
    print(best_result)
    print("-------------------")
    return best_result

def TreeClass():
    A_resultado, clf = Tree()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain = XaTrain[:,0:18]
    XaTest = XaTest[:,0:18]
    tree_class = best_model(clf, XaTrain, yaTrain)
    print(tree_class)
