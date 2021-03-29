import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.naive_bayes import GaussianNB

def NB():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = GaussianNB()
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ScoreNB():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    A_resultado, clf = NB()
    score_Train_A = clf.score(XaTrain,yaTrain)
    score_Test_A = clf.score(XaTest,yaTest)
