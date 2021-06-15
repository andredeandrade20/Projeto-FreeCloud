import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.neighbors import KNeighborsClassifier
import time
import evaluatemodel as evm
import seaborn as sns
import matplotlib.pyplot as plt

##Instânciação do classificador
def KNN():
    inicio = time.time()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    print("Treinando modelo....")
    clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
    clf = clf.fit(XaTrain, yaTrain)
    yPred = clf.predict(XaTest)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    return XaTrain, XaTest, yaTrain, yaTest, yPred, clf

def KNNMetrics():
    print("-------------------")
    print("Métricas K-Neirest Neighbors")
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = KNN()
    inicio = time.time()
    evm.CrossValidation(clf, XaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")

def PlotKNN():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = KNN()
    XaTest = pd.DataFrame(XaTest)
    yPred = pd.DataFrame(yPred)
    df = pd.concat([XaTest, yaTest], axis = 1)
    df.insert(27, 'Predições', yPred.values, allow_duplicates = False)
    sns.relplot(data=df, x="Banda 1", y='Nuvem Alta', hue = 'Predições')
    plt.show()
