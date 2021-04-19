import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.tree import DecisionTreeClassifier
import time
import evaluatemodel as evm

##Instânciação do classificador
def Tree():
    inicio = time.time()
    print("Treinando modelo....")
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
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
    evm.CrossValidation(clf, yaTest, yPred)
    final = time.time() - inicio
    min = final/60
    print('Tempo de Execução: {} min '.format(min))
    print('Final das Métricas')
    print("-------------------")
