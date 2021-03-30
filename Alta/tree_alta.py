import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix



def Tree():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best')
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ScoreTree():
    A_resultado, clf = Tree()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    score_Train_A = clf.score(XaTrain,yaTrain)
    score_Test_A = clf.score(XaTest,yaTest)
    score_resultado_A = clf.score(XaTest, A_resultado)
    print(score_Train_A, score_Test_A, score_resultado_A)
    return score_Train_A, score_Test_A, score_resultado_A

def ConfusionMatrixScore():
    A_resultado, clf = Tree()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    cm_A = confusion_matrix(yaTest, A_resultado)

    print(cm_A)


def GraphViz():
    B_resultado, clf = Tree()
    XbTrain, XbTest, ybTrain, ybTest = pre.TrainTestSet()
    export_graphviz(clf, out_file = 'tree_alta.dot')
