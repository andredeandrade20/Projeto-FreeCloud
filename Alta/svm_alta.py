import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.svm import LinearSVC
import confusionmatrix as cm

def SVC():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = LinearSVC(max_iter = 10000)
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixSVC():
    A_resultado, clf = SVC()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    ConfusionMatrixSVC = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixSVC

def EvaluateSVCClouds(ConfusionMatrixSVC):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixSVC)

def EvaluateSVCTotal(ConfusionMatrixSVC):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixSVC)
