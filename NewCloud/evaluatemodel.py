from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

## Validação cruzada
def CrossValidation(clf, X, y):
    scoring = ['acurracy', 'precision', 'recall']
    metricas = cross_val(clf, X, y, cv = 10, scoring = scoring)
    print(Métricas)


def CrossValTree():
    A_resultado, clf, XaTrain, XaTest, yaTrain, yaTest = Tree()
    cv.CrossValidation(clf, XaTrain, yaTrain)
    cv.CrossValidation(clf, XaTest, A_resultado)
