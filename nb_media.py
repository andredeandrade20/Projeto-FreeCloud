import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.naive_bayes import GaussianNB

def NB():
    XmTrain, XmTest, ymTrain, ymTest = pre.TrainTestMedia()
    clf = GaussianNB()
    clf = clf.fit(XmTrain, ymTrain)
    M_resultado = clf.predict(XmTest)
    print(M_resultado)
    return M_resultado, clf

def ScoreNB():
    XmTrain, XmTest, ymTrain, ymTest = pre.TrainTestMedia()
    M_resultado, clf = NB()
    score_Train_M = clf.score(XmTrain,ymTrain)
    score_Test_M = clf.score(XmTest,ymTest)
    print(score_Train_M, score_Test_M)
