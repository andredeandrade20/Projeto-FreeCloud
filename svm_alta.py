import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.svm import SVC

def SVC():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    clf = SVC(kernel = 'linear')
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    print(A_resultado)
    return A_resultado, clf

def ScoreSVC():
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    A_resultado, clf = SVC()
    score_Train_A = clf.score(XaTrain,yaTrain)
    score_Test_A = clf.score(XaTest,yaTest)
    print(score_Train_A, score_Test_A)
