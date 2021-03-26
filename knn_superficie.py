import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.neighbors import KNeighborsClassifier

def KNN():
    XsTrain, XsTest, ysTrain, ysTest = pre.TrainTestSuperficie()
    clf = KNeighborsClassifier(n_neighbors = 5, p = 2)
    clf = clf.fit(XsTrain, ysTrain)
    S_resultado = clf.predict(XsTest)
    print(S_resultado)
    return S_resultado, clf

def ScoreKNN():
    XsTrain, XsTest, ysTrain, ysTest = pre.TrainTestSuperficie()
    S_resultado, clf = KNN()
    score_Train_S = clf.score(XsTrain,ysTrain)
    score_Test_S = clf.score(XsTest,ysTest)
    print(score_Train_S, score_Test_S)
