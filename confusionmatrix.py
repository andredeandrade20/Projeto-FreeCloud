import tree_alta
import preprocessing as pre
from sklearn.metrics import multilabel_confusion_matrix
import nb_alta


def ConfusionMatrixScore():
    A_resultado, clf = nb_alta.NB()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    cm_A = multilabel_confusion_matrix(yaTest, A_resultado)
    print(cm_A)
