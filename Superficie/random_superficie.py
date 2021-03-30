import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.tree import RandomForestClassifier

def RandomTree():
    XsTrain, XsTest, ysTrain, ysTest = pre.TrainTestSuperficie()
    clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
    clf = clf.fit(XsTrain, ysTrain)
    S_resultado = clf.predict(XsTest)
    return S_resultado, clf

def ScoreRandomTree():
    S_resultado, clf = Tree()
    XsTrain, XsTest, ysTrain, ysTest = pre.TrainTestSuperficie()
    score_Train_S = clf.score(XsTrain,ysTrain)
    score_Test_S = clf.score(XsTest,ysTest)
    score_resultado_S = clf.score(XsTest,S_resultado)
    print(score_Train_S, score_Test_S, score_resultado_S)
    return score_Train_S, score_Test_S, score_resultado_S
