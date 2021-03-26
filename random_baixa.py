import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.tree import RandomForestClassifier
from sklearn import metrics

def RandomTree():
    XbTrain, XbTest, ybTrain, ybTest = pre.TrainTestBaixa()
    clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
    clf = clf.fit(XbTrain, ybTrain)
    B_resultado = clf.predict(XbTest)
    return B_resultado, clf

def ScoreRandomTree():
    B_resultado, clf = Tree()
    XbTrain, XbTest, ybTrain, ybTest = pre.TrainTestBaixa()
    score_Train_B = clf.score(XbTrain,ybTrain)
    score_Test_B = clf.score(XbTest,ybTest)
    score_resultado_B = clf.score(XbTest,B_resultado)
    print(score_Train_B, score_Test_B, score_resultado_B)
    return score_Train_B, score_Test_B, score_resultado_B
