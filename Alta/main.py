import preprocessing_alta as pre
import confusionmatrix as cm
import knn_alta as knn
import logisticregression_alta as lr
import mlp_alta as mlp
import nb_alta as nb
import random_alta as random
import svm_alta as svm
import tree_alta as tree

## Funções para rodar as métricas para K-Neighbors
## Função para rodar as métricas individuais de cada nuvem
def RunCloudsKNN():
    print("Métricas K-Neighbors para cada nuvem:")
    ConfusionMatrixKNN = knn.ConfusionMatrixKNN()
    knn.EvaluateKNNClouds(ConfusionMatrixKNN)
## Função para rodar as métricas gerais do modelo
def RunTotalKNN():
    print("\nMétricas K-Neighbors:")
    ConfusionMatrixKNN = knn.ConfusionMatrixKNN()
    knn.EvaluateKNNTotal(ConfusionMatrixKNN)

## Funções para a Regressão Logística
def RunCloudsLR():
    print("Métricas regressão logística para cada nuvem:")
    ConfusionMatrixLR = lr.ConfusionMatrixLR()
    lr.EvaluateLRClouds(ConfusionMatrixLR)

def RunTotalLR():
    print("\nMétricas regressão logística:")
    ConfusionMatrixLR = lr.ConfusionMatrixLR()
    lr.EvaluateLRTotal(ConfusionMatrixLR)

## Funções para o Perceptron de múltiplas camadas
def RunCloudsMLP():
    print("Métricas perceptron de múltiplas camadas para cada nuvem:")
    ConfusionMatrixLR = mlp.ConfusionMatrixMLP()
    mlp.EvaluateTrMLPClouds(ConfusionMatrixMLP)

def RunTotalMLP():
    print("\nMétricas perceptron de múltiplas camadas:")
    ConfusionMatrixMLP = mlp.ConfusionMatrixMLP()
    mlp.EvaluateMLPTotal(ConfusionMatrixMLP)

## Funções para o algoritmo de Naive Bayes
def RunCloudsNB():
    print("Métricas naive bayes para cada nuvem:")
    ConfusionMatrixNB = nb.ConfusionMatrixNB()
    nb.EvaluateNBClouds(ConfusionMatrixNB)

def RunTotalNB():
    print("\nMétricas naive bayes:")
    ConfusionMatrixNB = nb.ConfusionMatrixNB()
    nb.EvaluateNBTotal(ConfusionMatrixNB)

## Funções para Árvore de decisão
def RunCloudsSVC():
    print("Métricas Support Vector Machine (SVM) para cada nuvem:")
    ConfusionMatrixSVC = svm.ConfusionMatrixSVC()
    svm.EvaluateSVCClouds(ConfusionMatrixSVC)

def RunTotalSVC():
    print("\nMétricas Support Vector Machine (SVM):")
    ConfusionMatrixSVC = svm.ConfusionMatrixSVC()
    svm.EvaluateSVCTotal(ConfusionMatrixSVC)

## Funções para a floresta aleatória
def RunCloudsRandomTree():
    print("Métricas Random Forest para cada nuvem:")
    ConfusionMatrixRandomTree = random.ConfusionMatrixRandomTree()
    random.EvaluateRandomTreeClouds(ConfusionMatrixRandomTree)

def RunTotalRandomTree():
    print("\nMétricas Random Forest:")
    ConfusionMatrixRandomTree = random.ConfusionMatrixRandomTree()
    random.EvaluateRandomTreeTotal(ConfusionMatrixRandomTree)

## Funções para Árvore de decisão
def RunCloudsTree():
    print("Métricas árvore de decisão para cada nuvem:")
    ConfusionMatrixTree = tree.ConfusionMatrixTree()
    tree.EvaluateTreeClouds(ConfusionMatrixTree)

def RunTotalTree():
    print("\nMétricas árvore de decisão:")
    ConfusionMatrixTree = tree.ConfusionMatrixTree()
    tree.EvaluateTreeTotal(ConfusionMatrixTree)

def RunModels():
    RunTotalNB()
    RunTotalKNN()
    RunTotalTree()
    RunTotalRandomTree()
    RunTotalLR()
    RunTotalMLP()
    RunTotalSVC()

if __name__ == "__main__":
    mlp.plot()
