import pandas as pd
import numpy as np
import preprocessing as pre
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import confusion_matrix

def Tree():
    XsTrain, XsTest, ysTrain, ysTest = pre.TrainTestSuperficie()
    clf = DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', max_depth = 12)
    clf = clf.fit(XsTrain, ysTrain)
    S_resultado = clf.predict(XsTest)
    return S_resultado, clf

def ScoreTree():
    S_resultado, clf = Tree()
    XsTrain, XsTest, ysTrain, ysTest = pre.TrainTestSuperficie()
    score_Train_S = clf.score(XsTrain,ysTrain)
    score_Test_S = clf.score(XsTest,ysTest)
    score_resultado_S = clf.score(XsTest,S_resultado)
    print(score_Train_S, score_Test_S, score_resultado_S)
    return score_Train_S, score_Test_S, score_resultado_S

def GraphViz():
    S_resultado, clf = Tree()
    XsTrain, XsTest, ysTrain, ysTest = pre.TrainTestSuperficie()
    export_graphviz(clf, out_file = 'tree_superficie.dot')

def ConfusionMatrixScore():
    S_resultado, clf = Tree()
    XsTrain, XsTest, ysTrain, ysTest = pre.TrainTestSuperficie()
    confusionMatrix_S = confusion_matrix(ysTest, S_resultado)
    print(confusionMatrix_S)

## Não usadas
def Cross_Score():
    S_resultado, clf = Tree()
    XsTrain, XsTest, ysTrain, ysTest = pre.TrainTestSupuperficieficie()
    scores = cross_val_score(clf, XsTrain, XsTest, cv=5, scoring='accuracy')
    print(scores)

def TreeRegion():
    S_resultado, clf = Tree()
    XsTrain, XsTest, ysTrain, ysTest = pre.TrainTestSuperficie()
    plt.figure(figsize=(16,9))
    plot_decision_regions(XsTest[:,1:3].astype(np.integer),S_resultado.astype(np.integar), clf= clf, legend = 10)

    plt.xlabel('Canal 1')
    plt.ylabel('Nuvens')
    plt.title('Decision Tree')
    plt.show()
