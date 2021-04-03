import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.ensemble import RandomForestClassifier
import confusionmatrix as cm

def RandomTree():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixRandomTree():
    A_resultado, clf = RandomTree()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    ConfusionMatrixRandomTree = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixRandomTree

def EvaluateRandomTreeClouds(ConfusionMatrixRandomTree):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixRandomTree)

def EvaluateRandomTreeTotal(ConfusionMatrixRandomTree):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixRandomTree)
