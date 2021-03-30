import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.neighbors import KNeighborsClassifier

def KNN():
    XbTrain, XbTest, ybTrain, ybTest = pre.TrainTestBaixa()
    clf = KNeighborsClassifier(n_neighbors = 5, p = 2)
    clf = clf.fit(XbTrain, ybTrain)
    B_resultado = clf.predict(XbTest)
    print(B_resultado)
    return B_resultado, clf

def ScoreKNN():
    XbTrain, XbTest, ybTrain, ybTest = pre.TrainTestBaixa()
    B_resultado, clf = KNN()
    score_Train_B = clf.score(XbTrain,ybTrain)
    score_Test_B = clf.score(XbTest,ybTest)
    print(score_Train_B, score_Test_B)
