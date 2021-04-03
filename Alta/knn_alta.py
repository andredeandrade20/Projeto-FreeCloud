import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.neighbors import KNeighborsClassifier
import confusionmatrix as cm

def KNN():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = KNeighborsClassifier(n_neighbors = 5, p = 2)
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixKNN():
    A_resultado, clf = KNN()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    ConfusionMatrixKNN = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixKNN

def EvaluateKNNClouds(ConfusionMatrixKNN):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixKNN)

def EvaluateKNNTotal(ConfusionMatrixKNN):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixKNN)
