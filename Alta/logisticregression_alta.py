import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.linear_model import LogisticRegression
import confusionmatrix as cm

def LR():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter = 200)
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixLR():
    A_resultado, clf = LR()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    ConfusionMatrixLR = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixLR

def EvaluateLRClouds(ConfusionMatrixLR):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixLR)

def EvaluateLRTotal(ConfusionMatrixLR):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixLR)
