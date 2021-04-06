import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.neural_network import MLPClassifier
import confusionmatrix as cm
import matplotlib.pyplot as plt

def MLP():
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain = XaTrain[:,0:18]
    XaTest = XaTest[:,0:18]
    clf = MLPClassifier(max_iter = 800)
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixMLP():
    A_resultado, clf = MLP()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    ConfusionMatrixMLP = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixMLP

def EvaluateMLPClouds(ConfusionMatrixMLP):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixMLP)

def EvaluateMLPTotal(ConfusionMatrixMLP):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixMLP)

def plot():
    A_resultado, clf = MLP()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    plt.scatter(XaTest[:,-1], yaTest, color = 'red')
    plt.plot(XaTest[:,-1], A_resultado, color = 'blue')
    plt.xlabel("Tempo")
    plt.ylabel("Nuvem")
    plt.title("Nuvem x Tempo")
    plt.show()
