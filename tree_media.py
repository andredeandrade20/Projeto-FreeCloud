import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

def Tree():
    XmTrain, XmTest, ymTrain, ymTest = pre.TrainTestSet()
    clf = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', max_depth = 6)
    clf = clf.fit(XmTrain, ymTrain)
    M_resultado = clf.predict(XmTest)
    return M_resultado, clf

def ScoreTree():
    M_resultado, clf = Tree()
    XmTrain, XmTest, ymTrain, ymTest = pre.TrainTestSet()
    score_Train_M = clf.score(XmTrain,ymTrain)
    score_Test_M = clf.score(XmTest,ymTest)
    score_resultado_M = clf.score(XmTest,M_resultado)
    print(score_Train_M, score_Test_M, score_resultado_M)
    return score_Train_M, score_Test_M, score_resultado_M

def GraphViz():
    B_resultado, clf = Tree()
    XbTrain, XbTest, ybTrain, ybTest = pre.TrainTestSet()
    export_graphviz(clf, out_file = 'tree_media.dot')
