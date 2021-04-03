import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.naive_bayes import GaussianNB
import confusionmatrix as cm

def NB():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = GaussianNB()
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixNB():
    A_resultado, clf = NB()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    ConfusionMatrixNB = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixNB

def EvaluateNBClouds(ConfusionMatrixNB):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixNB)

def EvaluateNBTotal(ConfusionMatrixNB):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixNB)
