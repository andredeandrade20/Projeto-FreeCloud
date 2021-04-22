import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.tree import DecisionTreeClassifier
import time
import evaluatemodel as evm
import seaborn as sns
import matplotlib.pyplot as plt

##Instânciação do classificador
def Tree():
    inicio = time.time()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    XaTrain, XaTest = pre.Chnl(XaTrain, XaTest)
    print("Treinando modelo....")
    clf = DecisionTreeClassifier(max_depth=12, min_samples_leaf=3, min_samples_split=0.1)
    clf = clf.fit(XaTrain, yaTrain)
    yPred = clf.predict(XaTest)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    return XaTrain, XaTest, yaTrain, yaTest, yPred, clf

def TreeMetrics():
    print("-------------------")
    print("Métricas Árvore de decisão")
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = Tree()
    inicio = time.time()
    evm.CrossValidation(clf, XaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")

def PlotTree():
    XaTrain, XaTest, yaTrain, yaTest, yPred, clf = Tree()
    XaTest = pd.DataFrame(XaTest)
    yPred = pd.DataFrame(yPred)
    df = pd.concat([XaTest, yaTest], axis = 1)
    df.insert(27, "Predições", yPred.values, allow_duplicates = False )
    sns.catplot(x="Banda 1", y="Nuvem Alta", data=df)
    plt.show()
