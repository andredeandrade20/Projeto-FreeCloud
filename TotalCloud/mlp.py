import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import time
import evaluatemodel as evm
import seaborn as sns

def MLP():
    inicio = time.time()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    print("Treinando modelo....")
    clf = MLPClassifier(max_iter = 12000, hidden_layer_sizes = (1000,500,250))
    clf = clf.fit(XaTrain, yaTrain)
    yPred = clf.predict(XaTest)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    return XaTrain, XaTest, yaTrain, yaTest, yPred, clf

def MLPMetrics():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = MLP()
    print("-------------------")
    print("Métricas Percéptron de Multiplas camadas")
    inicio = time.time()
    evm.CrossValidation(clf, XaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")

def PlotMLP():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = MLP()
    XaTest = pd.DataFrame(XaTest)
    yPred = pd.DataFrame(yPred)
    df = pd.concat([XaTest, yaTest], axis = 1)
    df.insert(27, 'Predições', yPred.values, allow_duplicates = False)
    sns.relplot(data=df, x="Banda 1", y='Nuvem Alta', hue = 'Predições')
    plt.show()
