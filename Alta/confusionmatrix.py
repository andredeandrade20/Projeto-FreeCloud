import tree_alta
import preprocessing as pre
from sklearn.metrics import multilabel_confusion_matrix
import nb_alta

def ConfusionMatrix():
    A_resultado, clf = nb_alta.NB()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    cm_A = multilabel_confusion_matrix(yaTest, A_resultado)
    return A_resultado, clf, XaTrain, XaTest, yaTrain, yaTest, cm_A

def ConfusionMatrixScore():

    A_resultado, clf = nb_alta.NB()
    XaTrain, XaTest, yaTrain, yaTest = pre.TrainTestAlta()
    cm_A = multilabel_confusion_matrix(yaTest, A_resultado)

    cm_1 = cm_A[[0][0]]
    cm_1_acc = (cm_1[0,0] + cm_1[1,1])/(cm_1[0,0] + cm_1[0,1] + cm_1[1,0] + cm_1[1,1])
    cm_1_recall = (cm_1[0,0])/(cm_1[0,0] + cm_1[0,1])
    cm_1_prec = (cm_1[0,0])/(cm_1[0,0] + cm_1[1,0])
    cm_1_f1 = (2*cm_1_prec*cm_1_recall)/(cm_1_prec + cm_1_recall)

    cm_2 = cm_A[[1][0]]
    cm_2_acc = (cm_2[0,0] + cm_2[1,1])/(cm_2[0,0] + cm_2[0,1] + cm_2[1,0] + cm_2[1,1])
    cm_2_recall = (cm_2[0,0])/(cm_2[0,0] + cm_2[0,1])
    cm_2_prec = (cm_2[0,0])/(cm_2[0,0] + cm_2[1,0])
    cm_2_f1 = (2*cm_2_prec*cm_2_recall)/(cm_2_prec + cm_2_recall)

    cm_3 = cm_A[[2][0]]
    cm_3_acc = (cm_3[0,0] + cm_3[1,1])/(cm_3[0,0] + cm_3[0,1] + cm_3[1,0] + cm_3[1,1])
    cm_3_recall = (cm_3[0,0])/(cm_3[0,0] + cm_3[0,1])
    cm_3_prec = (cm_3[0,0])/(cm_3[0,0] + cm_3[1,0])
    cm_3_f1 = (2*cm_3_prec*cm_3_recall)/(cm_3_prec + cm_3_recall)

    cm_4 = cm_A[[3][0]]
    cm_4_acc = (cm_1[0,0] + cm_1[1,1])/(cm_1[0,0] + cm_1[0,1] + cm_1[1,0] + cm_1[1,1])
    cm_4_recall = (cm_1[0,0])/(cm_1[0,0] + cm_1[0,1])
    cm_4_prec = (cm_1[0,0])/(cm_1[0,0] + cm_1[1,0])
    cm_4_f1 = (2*cm_1_prec*cm_1_recall)/(cm_1_prec + cm_1_recall)

    cm_5 = cm_A[[4][0]]
    cm_5_acc = (cm_1[0,0] + cm_1[1,1])/(cm_1[0,0] + cm_1[0,1] + cm_1[1,0] + cm_1[1,1])
    cm_5_recall = (cm_1[0,0])/(cm_1[0,0] + cm_1[0,1])
    cm_5_prec = (cm_1[0,0])/(cm_1[0,0] + cm_1[1,0])
    cm_5_f1 = (2*cm_1_prec*cm_1_recall)/(cm_1_prec + cm_1_recall)

    cm_6 = cm_A[[5][0]]
    cm_6_acc = (cm_1[0,0] + cm_1[1,1])/(cm_1[0,0] + cm_1[0,1] + cm_1[1,0] + cm_1[1,1])
    cm_6_recall = (cm_1[0,0])/(cm_1[0,0] + cm_1[0,1])
    cm_6_prec = (cm_1[0,0])/(cm_1[0,0] + cm_1[1,0])
    cm_6_f1 = (2*cm_1_prec*cm_1_recall)/(cm_1_prec + cm_1_recall)

    cm_7 = cm_A[[6][0]]
    cm_7_acc = (cm_1[0,0] + cm_1[1,1])/(cm_1[0,0] + cm_1[0,1] + cm_1[1,0] + cm_1[1,1])
    cm_7_recall = (cm_1[0,0])/(cm_1[0,0] + cm_1[0,1])
    cm_7_prec = (cm_1[0,0])/(cm_1[0,0] + cm_1[1,0])
    cm_7_f1 = (2*cm_1_prec*cm_1_recall)/(cm_1_prec + cm_1_recall)

    cm_8 = cm_A[[7][0]]
    cm_8_acc = (cm_1[0,0] + cm_1[1,1])/(cm_1[0,0] + cm_1[0,1] + cm_1[1,0] + cm_1[1,1])
    cm_8_recall = (cm_1[0,0])/(cm_1[0,0] + cm_1[0,1])
    cm_8_prec = (cm_1[0,0])/(cm_1[0,0] + cm_1[1,0])
    cm_2_f1 = (2*cm_1_prec*cm_1_recall)/(cm_1_prec + cm_1_recall)

    cm_9 = cm_A[[8][0]]
    cm_9_acc = (cm_1[0,0] + cm_1[1,1])/(cm_1[0,0] + cm_1[0,1] + cm_1[1,0] + cm_1[1,1])
    cm_9_recall = (cm_1[0,0])/(cm_1[0,0] + cm_1[0,1])
    cm_9_prec = (cm_1[0,0])/(cm_1[0,0] + cm_1[1,0])
    cm_9_f1 = (2*cm_1_prec*cm_1_recall)/(cm_1_prec + cm_1_recall)

    print(cm_1_acc,cm_1_recall,cm_1_prec, cm_1_f1)
