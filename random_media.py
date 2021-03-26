import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.tree import RandomForestClassifier
from sklearn import metrics

def RandomTree():
    XmTrain, XmTest, ymTrain, ymTest = pre.TrainTestBaixa()
    clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
    clf = clf.fit(XmTrain, ymTrain)
    M_resultado = clf.predict(XmTest)
    return M_resultado, clf

def ScoreRandomTree():
    M_resultado, clf = Tree()
    XmTrain, XmTest, ymTrain, ymTest = pre.TrainTestBaixa()
    score_Train_M = clf.score(XmTrain,ymTrain)
    score_Test_M = clf.score(XmTest,ymTest)
    score_resultado_M = clf.score(XmTest,M_resultado)
    print(score_Train_M, score_Test_M, score_resultado_M)
    return score_Train_M, score_Test_M, score_resultado_M
