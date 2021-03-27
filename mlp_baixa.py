import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.neural_network import MLPClassifier

def MLP():
    XbTrain, XbTest, ybTrain, ybTest = pre.TrainTestBaixa()
    clf = MLPClassifier()
    clf = clf.fit(XbTrain, ybTrain)
    B_resultado = clf.predict(XbTest)
    print(B_resultado)
    return B_resultado, clf

def ScoreMLP():
    XbTrain, XbTest, ybTrain, ybTest = pre.TrainTestBaixa()
    B_resultado, clf = MLP()
    score_Train_B = clf.score(XbTrain,ybTrain)
    score_Test_B = clf.score(XbTest,ybTest)
    print(score_Train_B, score_Test_B)
