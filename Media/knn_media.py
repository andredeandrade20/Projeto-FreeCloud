import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.neighbors import KNeighborsClassifier

def KNN():
    XmTrain, XmTest, ymTrain, ymTest = pre.TrainTestMedia()
    clf = KNeighborsClassifier(n_neighbors = 5, p = 2)
    clf = clf.fit(XmTrain, ymTrain)
    M_resultado = clf.predict(XmTest)
    print(M_resultado)
    return M_resultado, clf

def ScoreKNN():
    XmTrain, XmTest, ymTrain, ymTest = pre.TrainTestMedia()
    M_resultado, clf = KNN()
    score_Train_M = clf.score(XmTrain,ymTrain)
    score_Test_M = clf.score(XmTest,ymTest)
    print(score_Train_M, score_Test_M)
