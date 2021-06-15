import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.naive_bayes import GaussianNB
import time
import evaluatemodel as evm
import seaborn as sns
import matplotlib.pyplot as plt

##Instânciação do classificador
def NB():
    inicio = time.time()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    print("Treinando modelo....")
    clf = GaussianNB(priors = None, var_smoothing = 1e-09)
    clf = clf.fit(XaTrain, yaTrain)
    yPred = clf.predict(XaTest)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    return XaTrain, XaTest, yaTrain, yaTest, yPred, clf

def NBMetrics():
    print("-------------------")
    print("Métricas Naive Bayes")
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = NB()
    inicio = time.time()
    evm.CrossValidation(clf, XaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")

def PlotTree():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = NB()
    XaTest = pd.DataFrame(XaTest)
    yPred = pd.DataFrame(yPred)
    df = pd.concat([XaTest, yaTest], axis = 1)
    df.insert(27, 'Predições', yPred.values, allow_duplicates = False)
    sns.relplot(data=df, x="Banda 1", y='Nuvem Alta', hue = 'Predições')
    plt.show()
