import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

def Tree():
    XbTrain, XbTest, ybTrain, ybTest = pre.TrainTestSet()
    clf = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', max_depth = 15)
    clf = clf.fit(XbTrain, ybTrain)
    B_resultado = clf.predict(XbTest)
    return B_resultado, clf

def ScoreTree():
    B_resultado, clf = Tree()
    XbTrain, XbTest, ybTrain, ybTest = pre.TrainTestSet()
    score_Train_B = clf.score(XbTrain,ybTrain)
    score_Test_B = clf.score(XbTest,ybTest)
    score_resultado_B = clf.score(XbTest,B_resultado)
    print(score_Train_B, score_Test_B, score_resultado_B)
    return score_Train_B, score_Test_B, score_resultado_B

def GraphViz():
    B_resultado, clf = Tree()
    XbTrain, XbTest, ybTrain, ybTest = pre.TrainTestSet()
    export_graphviz(clf, out_file = 'tree_baixa.dot')
