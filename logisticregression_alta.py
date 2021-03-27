import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.neural_network import LogisticRegression

def LR():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = LogisticRegression()
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    print(A_resultado)
    return A_resultado, clf

def ScoreLR():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    A_resultado, clf = LR()
    score_Train_A = clf.score(XaTrain,yaTrain)
    score_Test_A = clf.score(XaTest,yaTest)
    print(score_Train_A, score_Test_A)
