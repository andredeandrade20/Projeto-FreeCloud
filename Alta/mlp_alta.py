import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.neural_network import MLPClassifier
import confusionmatrix as cm

def MLP():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = MLPClassifier(max_iter = 600)
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixMLP():
    A_resultado, clf = MLP()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    ConfusionMatrixMLP = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixMLP

def EvaluateMLPClouds(ConfusionMatrixMLP):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixMLP)

def EvaluateMLPTotal(ConfusionMatrixMLP):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixMLP)
