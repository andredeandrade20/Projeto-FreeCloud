import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.tree import DecisionTreeClassifier
import confusionmatrix as cm

def Tree():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best')
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixTree():
    A_resultado, clf = Tree()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    ConfusionMatrixTree = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixTree

def EvaluateTreeClouds(ConfusionMatrixTree):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixTree)

def EvaluateTreeTotal(ConfusionMatrixTree):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixTree)
