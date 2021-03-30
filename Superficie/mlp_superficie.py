import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.neural_network import MLPClassifier

def MLP():
    XsTrain, XsTest, ysTrain, ysTest = pre.TrainTestSuperficie()
    clf = MLPClassifier()
    clf = clf.fit(XsTrain, ysTrain)
    S_resultado = clf.predict(XsTest)
    print(S_resultado)
    return S_resultado, clf

def ScoreMLP():
    XsTrain, XsTest, ysTrain, ysTest = pre.TrainTestSuperficie()
    S_resultado, clf = MLP()
    score_Train_S = clf.score(XsTrain,ysTrain)
    score_Test_S = clf.score(XsTest,ysTest)
    print(score_Train_S, score_Test_S)
