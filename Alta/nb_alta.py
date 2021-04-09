import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.naive_bayes import GaussianNB
import confusionmatrix as cm
import cross_validation as cv

def NB():
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain = XaTrain[:,0:18]
    XaTest = XaTest[:,0:18]
    clf = GaussianNB()
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixNB():
    A_resultado, clf = NB()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    ConfusionMatrixNB = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixNB

def EvaluateNBClouds(ConfusionMatrixNB):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixNB)

def EvaluateNBTotal(ConfusionMatrixNB):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixNB)

def CrossValNB():
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    A_resultado, clf = NB()
    cv.CrossValidation(clf, XaTest[:,0:18], yaTest)
