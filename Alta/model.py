import pandas as pd
import numpy as np
import preprocessing_alta as pre
from sklearn.svm import LinearSVC
import confusionmatrix as cm

def Model(clf):
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainData()
    clf = LinearSVC(max_iter = 10000)
    clf = clf.fit(XaTrain, yaTrain)
    A_resultado = clf.predict(XaTest)
    return A_resultado, clf

def ConfusionMatrixSVC():
    A_resultado, clf = SVC()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestData()
    ConfusionMatrixSVC = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixSVC

### Funções para utilização da validação cruzada ###

''' Estas funções são utilizadas para realizar o método de validação cruzada através do GridSearchCV
      a fim de encontrar os melhores hiperparâmetros do modelo de Machine Learning '''


def config_param(clf, param, cv= None, n_jobs=-1, scoring = 'balanced_accuracy'):

    grid_class = GridSearchCV(clf, param, cv=cv,n_jobs=n_jobs, scoring = scoring)

    return clf, grid_class

def get_param(MLmodel, param, X,Y):

    MLmodel, grid_class = config_param(MLmodel, param)

    return grid_class.fit(X,Y)


def best_model(MLmodel,data_X,data_y,):
  print('----------------')
  print('Início do CVGrid')
  inicio = time.time()
  all_param = get_param(MLmodel, param, data_X,data_y)
  best_result = all_param.best_estimator_
  final = time.time() - inicio
  min = final/60
  print('Final do CVGrid')
  print('Tempo de Execução: {} min '.format(min))
  print('----------------')
  return best_result


MLModel = DecisionTreeClassifier()
tree_class = best_model(MLmodel, full_data_X_train,y_train_cancer)
print(tree_class)


def EvaluateSVCClouds(ConfusionMatrixSVC):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixSVC)

def EvaluateSVCTotal(ConfusionMatrixSVC):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixSVC)
