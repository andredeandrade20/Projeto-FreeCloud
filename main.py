import pandas as pd
import numpy as np
import knn_alta
import knn_media
import knn_baixa
import knn_superficie
import tree_alta
import tree_media
import tree_baixa
import tree_superficie
import nb_alta
import nb_media
import nb_baixa
import nb_superficie
import svm_alta
import svm_media
import svm_baixa
import svm_superficie
import preprocessing as pre
import confusionmatrix as cm

def RunScore():
    a = tree_superficie.ScoreTree()
    b = tree_baixa.ScoreTree()
    c = tree_media.ScoreTree()
    d = tree_alta.ScoreTree()


def RunResultados():
    S_resultado = tree_superficie.Tree()
    B_resultado = tree_baixa.Tree()
    M_resultado = tree_media.Tree()
    A_resultado = tree_alta.Tree()
    print("Resultados para superfície", S_resultado)
    print("Resultados para nuvens baixas", B_resultado)
    print("Resultados para nuvens médias", M_resultado)
    print("Resultados para nuvens altas", A_resultado)


if __name__ == "__main__":
    cm.ConfusionMatrixScore()
    tree_alta.ScoreTree()
