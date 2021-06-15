import preprocessing as pre
import knn
import lr
import mlp
import nb
import randomforest
import svm
import tree

def RunMetrics():
    print(".................")
    print("Qual modelo?:")
    x = str(input())
    if x == "k-neirest neighbors":
        knn.KNNMetrics()
    if x == "logistic regression":
        lr.LRMetrics()
    if x == "multi-layer perceptron":
        mlp.MLPMetrics()
    if x == "naive bayes":
        nb.NBMetrics()
    if x == "random forest":
        randomforest.RandomForestMetrics()
    if x == "support vector machine":
        svm.SVMMetrics()
    if x == "decision tree":
        tree.TreeMetrics()
    pass

if __name__ == "__main__":
    RunMetrics()
