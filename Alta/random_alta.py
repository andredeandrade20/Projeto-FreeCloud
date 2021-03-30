import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def RandomTree():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ScoreRandomTree():
    A_resultado, clf = Tree()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    score_Train_A = clf.score(XaTrain,yaTrain)
    score_Test_A = clf.score(XaTest,yaTest)
    score_resultado_A = clf.score(XaTest, A_resultado)
    print(score_Train_A, score_Test_A, score_resultado_A)
    return score_Train_A, score_Test_A, score_resultado_A
