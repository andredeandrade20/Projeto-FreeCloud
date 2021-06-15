import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.svm import LinearSVC
import time
import evaluatemodel as evm
import seaborn as sns
import matplotlib.pyplot as plt

##Instânciação do classificador
def SVM():
    inicio = time.time()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    print("Treinando modelo....")
    clf = LinearSVC(max_iter = 5000, class_weight = 'balanced', multi_class = 'crammer_singer')
    clf = clf.fit(XaTrain, yaTrain)
    yPred = clf.predict(XaTest)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    return XaTrain, XaTest, yaTrain, yaTest, yPred, clf

def SVMMetrics():
    print("-------------------")
    print("Métricas Support Vector Machine")
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = SVM()
    inicio = time.time()
    evm.CrossValidation(clf, XaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")

def PlotSVM():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = SVM()
    XaTest = pd.DataFrame(XaTest)
    yPred = pd.DataFrame(yPred)
    df = pd.concat([XaTest, yaTest], axis = 1)
    df.insert(27, 'Predições', yPred.values, allow_duplicates = False)
    sns.relplot(data=df, x="Banda 1", y='Nuvem Alta', hue = 'Predições')
    plt.show()
