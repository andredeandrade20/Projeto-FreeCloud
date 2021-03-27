import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.linear_model import LogisticRegression

def LR():
    XbTrain, XbTest, ybTrain, ybTest = pre.TrainTestBaixa()
    clf = LogisticRegression()
    clf = clf.fit(XbTrain, ybTrain)
    B_resultado = clf.predict(XbTest)
    print(B_resultado)
    return B_resultado, clf

def ScoreLR():
    XbTrain, XbTest, ybTrain, ybTest = pre.TrainTestBaixa()
    B_resultado, clf = LR()
    score_Train_B = clf.score(XbTrain,ybTrain)
    score_Test_B = clf.score(XbTest,ybTest)
    print(score_Train_B, score_Test_B)
