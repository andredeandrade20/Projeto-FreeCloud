import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.neural_network import MLPClassifier

def MLP():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = MLPClassifier()
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    print(A_resultado)
    return A_resultado, clf

def ScoreMLP():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    A_resultado, clf = MLP()
    score_Train_A = clf.score(XaTrain,yaTrain)
    score_Test_A = clf.score(XaTest,yaTest)
    print(score_Train_A, score_Test_A)
