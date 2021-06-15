from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

## Validação cruzada K-Fold que separa os conjuntos de treino e teste e tira as métricas médias
def CrossValidation(clf, X, y):
    acc = cross_val_score(clf, X, y, cv = 10, scoring = 'balanced_accuracy', error_score = 'raise')
    prec = cross_val_score(clf, X, y, cv = 10, scoring = 'precision_weighted', error_score = 'raise')
    rec = cross_val_score(clf, X, y, cv = 10, scoring = 'recall_weighted', error_score = 'raise')
    f1 = cross_val_score(clf, X, y, cv = 10, scoring = 'f1_macro', error_score = 'raise')
    roc = cross_val_score(clf, X, y, cv = 10, scoring = 'roc_auc_ovr', error_score = 'raise')
    print("Acurácia: ", acc.mean())
    print("Precisão: ", prec.mean())
    print("Recall: ", rec.mean())
    print("F1-Score: ", f1.mean())
    print("ROC-AUC: ", roc.mean())

def ConfusionMatrix(clf, XTest, YTest):
    plot_confusion_matrix(clf, XTest, YTest)
    plt.savefig('cm.png')
