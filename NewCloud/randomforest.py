import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.ensemble import RandomForestClassifier
import time
import evaluatemodel as evm
import seaborn as sns
import matplotlib.pyplot as plt

def RandomForest():
    inicio = time.time()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain, XaTest = pre.Chnl(XaTrain, XaTest)
    print("Treinando modelo....")
    clf = RandomForestClassifier()
    clf = clf.fit(XaTrain, yaTrain)
    yPred = clf.predict(XaTest)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    return XaTrain, XaTest, yaTrain, yaTest, yPred, clf

def RandomForestMetrics():
    print("-------------------")
    print("Métricas Random Forest")
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = RandomForest()
    inicio = time.time()
    evm.CrossValidation(clf, XaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")

def PlotRandomForest():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = RandomForest()
    XaTest = pd.DataFrame(XaTest)
    yPred = pd.DataFrame(yPred)
    df = pd.concat([XaTest, yaTest], axis = 1)
    df.insert(27, 'Predições', yPred.values, allow_duplicates = False)
    df.insert(28, 'Banda', df.iloc[:,0], allow_duplicates = False)
    sns.relplot(data=df, x="Banda", y='Nuvem Alta', hue = 'Predições')
    plt.show()

def ConfusionRandomForest():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = RandomForest()
    evm.ConfusionMatrix(clf, XaTest, yaTest)
