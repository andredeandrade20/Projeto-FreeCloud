import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.neighbors import KNeighborsClassifier

def KNN():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = KNeighborsClassifier(n_neighbors = 5, p = 2)
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    print(A_resultado)
    return A_resultado, clf

def ScoreKNN():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    A_resultado, clf = KNN()
    score_Train_A = clf.score(XaTrain,yaTrain)
    score_Test_A = clf.score(XaTest,yaTest)
    print(score_Train_A, score_Test_A)
