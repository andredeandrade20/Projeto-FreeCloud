import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.svm import SVC
import time
import evaluatemodel as evm
import seaborn as sns
import matplotlib.pyplot as plt

##Instânciação do classificador
def SVM():
    inicio = time.time()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain, XaTest = pre.Chnl(XaTrain, XaTest)
    print("Treinando modelo....")
    clf = SVC(max_iter = 10000, class_weight = 'balanced', probability = True)
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
    df.insert(28, 'Banda', df.iloc[:,0], allow_duplicates = False)
    sns.relplot(data=df, x="Banda", y='Nuvem Alta', hue = 'Predições')
    plt.show()

def ConfusionSVM():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = SVM()
    evm.ConfusionMatrix(clf, XaTest, yaTest)
