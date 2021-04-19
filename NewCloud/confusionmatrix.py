from sklearn.metrics import multilabel_confusion_matrix

def ConfusionMatrix(yaTest, A_resultado):
    ConfusionMatrix = multilabel_confusion_matrix(yaTest, A_resultado)
    return ConfusionMatrix

def ConfusionMatrixCompute(ConfusionMatrix):

## Métricas para o tipo de nuvem 1
    cm_1 = ConfusionMatrix[[0][0]]
    cm_1_acc = (cm_1[0,0] + cm_1[1,1])/(cm_1[0,0] + cm_1[0,1] + cm_1[1,0] + cm_1[1,1])
    cm_1_recall = (cm_1[0,0])/(cm_1[0,0] + cm_1[0,1])
    cm_1_prec = (cm_1[0,0])/(cm_1[0,0] + cm_1[1,0])
    cm_1_f1 = (2*cm_1_prec*cm_1_recall)/(cm_1_prec + cm_1_recall)

## Métricas para o tipo de nuvem 2
    cm_2 = ConfusionMatrix[[1][0]]
    cm_2_acc = (cm_2[0,0] + cm_2[1,1])/(cm_2[0,0] + cm_2[0,1] + cm_2[1,0] + cm_2[1,1])
    cm_2_recall = (cm_2[0,0])/(cm_2[0,0] + cm_2[0,1])
    cm_2_prec = (cm_2[0,0])/(cm_2[0,0] + cm_2[1,0])
    cm_2_f1 = (2*cm_2_prec*cm_2_recall)/(cm_2_prec + cm_2_recall)

## Métricas para o tipo de nuvem 3
    cm_3 = ConfusionMatrix[[2][0]]
    cm_3_acc = (cm_3[0,0] + cm_3[1,1])/(cm_3[0,0] + cm_3[0,1] + cm_3[1,0] + cm_3[1,1])
    cm_3_recall = (cm_3[0,0])/(cm_3[0,0] + cm_3[0,1])
    cm_3_prec = (cm_3[0,0])/(cm_3[0,0] + cm_3[1,0])
    cm_3_f1 = (2*cm_3_prec*cm_3_recall)/(cm_3_prec + cm_3_recall)

## Métricas para o tipo de nuvem 4
    cm_4 = ConfusionMatrix[[3][0]]
    cm_4_acc = (cm_4[0,0] + cm_4[1,1])/(cm_4[0,0] + cm_4[0,1] + cm_4[1,0] + cm_4[1,1])
    cm_4_recall = (cm_4[0,0])/(cm_4[0,0] + cm_4[0,1])
    cm_4_prec = (cm_4[0,0])/(cm_4[0,0] + cm_4[1,0])
    cm_4_f1 = (2*cm_4_prec*cm_4_recall)/(cm_4_prec + cm_4_recall)

## Métricas para o tipo de nuvem 5
    cm_5 = ConfusionMatrix[[4][0]]
    cm_5_acc = (cm_5[0,0] + cm_5[1,1])/(cm_5[0,0] + cm_5[0,1] + cm_5[1,0] + cm_1[1,1])
    cm_5_recall = (cm_5[0,0])/(cm_5[0,0] + cm_5[0,1])
    cm_5_prec = (cm_5[0,0])/(cm_5[0,0] + cm_5[1,0])
    cm_5_f1 = (2*cm_5_prec*cm_5_recall)/(cm_5_prec + cm_5_recall)

## Métricas para o tipo de nuvem 6
    cm_6 = ConfusionMatrix[[5][0]]
    cm_6_acc = (cm_6[0,0] + cm_6[1,1])/(cm_6[0,0] + cm_6[0,1] + cm_6[1,0] + cm_6[1,1])
    cm_6_recall = (cm_6[0,0])/(cm_6[0,0] + cm_6[0,1])
    cm_6_prec = (cm_6[0,0])/(cm_6[0,0] + cm_6[1,0])
    cm_6_f1 = (2*cm_6_prec*cm_6_recall)/(cm_6_prec + cm_6_recall)

## Métricas para o tipo de nuvem 7
    cm_7 = ConfusionMatrix[[6][0]]
    cm_7_acc = (cm_7[0,0] + cm_7[1,1])/(cm_7[0,0] + cm_7[0,1] + cm_7[1,0] + cm_7[1,1])
    cm_7_recall = (cm_7[0,0])/(cm_7[0,0] + cm_7[0,1])
    cm_7_prec = (cm_7[0,0])/(cm_7[0,0] + cm_7[1,0])
    cm_7_f1 = (2*cm_7_prec*cm_7_recall)/(cm_7_prec + cm_7_recall)

## Métricas para o tipo de nuvem 8
    cm_8 = ConfusionMatrix[[7][0]]
    cm_8_acc = (cm_8[0,0] + cm_8[1,1])/(cm_8[0,0] + cm_8[0,1] + cm_8[1,0] + cm_8[1,1])
    cm_8_recall = (cm_8[0,0])/(cm_8[0,0] + cm_8[0,1])
    cm_8_prec = (cm_8[0,0])/(cm_8[0,0] + cm_8[1,0])
    cm_8_f1 = (2*cm_8_prec*cm_8_recall)/(cm_8_prec + cm_8_recall)

## Métricas para o tipo de nuvem 9
    cm_9 = ConfusionMatrix[[8][0]]
    cm_9_acc = (cm_9[0,0] + cm_9[1,1])/(cm_9[0,0] + cm_9[0,1] + cm_9[1,0] + cm_1[1,1])
    cm_9_recall = (cm_9[0,0])/(cm_9[0,0] + cm_9[0,1])
    cm_9_prec = (cm_9[0,0])/(cm_9[0,0] + cm_9[1,0])
    cm_9_f1 = (2*cm_9_prec*cm_9_recall)/(cm_9_prec + cm_9_recall)

## Métricas gerais
    VP = cm_1[0,0] + cm_2[0,0] + cm_3[0,0] + cm_4[0,0] + cm_5[0,0] + cm_6[0,0] + cm_7[0,0] + cm_8[0,0] + cm_9[0,0]
    VN = cm_1[1,1] + cm_2[1,1] + cm_3[1,1] + cm_4[1,1] + cm_5[1,1] + cm_6[1,1] + cm_7[1,1] + cm_8[1,1] + cm_9[1,1]
    FP = cm_1[1,0] + cm_2[1,0] + cm_3[1,0] + cm_4[1,0] + cm_5[1,0] + cm_6[1,0] + cm_7[1,0] + cm_8[1,0] + cm_9[1,0]
    FN = cm_1[0,1] + cm_2[0,1] + cm_3[0,1] + cm_4[0,1] + cm_5[0,1] + cm_6[0,1] + cm_7[0,1] + cm_8[0,1] + cm_9[0,1]
    cm_total_acc = (VP + VN)/(VP + FN + FP + VN)
    cm_total_recall = (VP)/(VP + FN)
    cm_total_prec = (VP)/(VP + FP)
    cm_total_f1 = (2*cm_total_prec*cm_total_recall)/(cm_total_prec + cm_total_recall)

    return cm_1_acc, cm_1_recall, cm_1_prec, cm_1_f1, cm_2_acc, cm_2_recall, cm_2_prec, cm_2_f1, cm_3_acc, cm_3_recall, cm_3_prec, cm_3_f1, cm_4_acc, cm_4_recall, cm_4_prec, cm_4_f1,  cm_5_acc, cm_5_recall, cm_5_prec, cm_5_f1,  cm_6_acc, cm_6_recall, cm_6_prec, cm_6_f1,  cm_7_acc, cm_7_recall, cm_7_prec, cm_7_f1,  cm_8_acc, cm_8_recall, cm_8_prec, cm_8_f1,  cm_9_acc, cm_9_recall, cm_9_prec, cm_9_f1,  cm_total_acc, cm_total_recall, cm_total_prec, cm_total_f1

def ConfusionMatrixScoreClouds(ConfusionMatrix):
    cm_1_acc, cm_1_recall, cm_1_prec, cm_1_f1, cm_2_acc, cm_2_recall, cm_2_prec, cm_2_f1, cm_3_acc, cm_3_recall, cm_3_prec, cm_3_f1, cm_4_acc, cm_4_recall, cm_4_prec, cm_4_f1,  cm_5_acc, cm_5_recall, cm_5_prec, cm_5_f1,  cm_6_acc, cm_6_recall, cm_6_prec, cm_6_f1,  cm_7_acc, cm_7_recall, cm_7_prec, cm_7_f1,  cm_8_acc, cm_8_recall, cm_8_prec, cm_8_f1,  cm_9_acc, cm_9_recall, cm_9_prec, cm_9_f1,  cm_total_acc, cm_total_recall, cm_total_prec, cm_total_f1 = ConfusionMatrixCompute(ConfusionMatrix)
    print('Métricas tipo de nuvem 1:' '\nAcurácia:', cm_1_acc, '\nRecall:', cm_1_recall, '\nPrecisão:', cm_1_prec, '\nF1_score:', cm_1_f1)
    print('\nMétricas tipo de nuvem 2:' '\nAcurácia:', cm_2_acc, '\nRecall:', cm_2_recall, '\nPrecisão:', cm_2_prec, '\nF1_score:', cm_2_f1)
    print('\nMétricas tipo de nuvem 3:' '\nAcurácia:', cm_3_acc, '\nRecall:', cm_3_recall, '\nPrecisão:', cm_3_prec, '\nF1_score:', cm_3_f1)
    print('\nMétricas tipo de nuvem 4:' '\nAcurácia:', cm_4_acc, '\nRecall:', cm_4_recall, '\nPrecisão:', cm_4_prec, '\nF1_score:', cm_4_f1)
    print('\nMétricas tipo de nuvem 5:' '\nAcurácia:', cm_5_acc, '\nRecall:', cm_5_recall, '\nPrecisão:', cm_5_prec, '\nF1_score:', cm_5_f1)
    print('\nMétricas tipo de nuvem 5:' '\nAcurácia:', cm_5_acc, '\nRecall:', cm_5_recall, '\nPrecisão:', cm_5_prec, '\nF1_score:', cm_5_f1)
    print('\nMétricas tipo de nuvem 7:' '\nAcurácia:', cm_7_acc, '\nRecall:', cm_7_recall, '\nPrecisão:', cm_7_prec, '\nF1_score:', cm_7_f1)
    print('\nMétricas tipo de nuvem 8:' '\nAcurácia:', cm_8_acc, '\nRecall:', cm_8_recall, '\nPrecisão:', cm_8_prec, '\nF1_score:', cm_8_f1)
    print('\nMétricas tipo de nuvem 9:' '\nAcurácia:', cm_9_acc, '\nRecall:', cm_9_recall, '\nPrecisão:', cm_9_prec, '\nF1_score:', cm_9_f1)


def ConfusionMatrixScoreTotal(ConfusionMatrix):
    cm_1_acc, cm_1_recall, cm_1_prec, cm_1_f1, cm_2_acc, cm_2_recall, cm_2_prec, cm_2_f1, cm_3_acc, cm_3_recall, cm_3_prec, cm_3_f1, cm_4_acc, cm_4_recall, cm_4_prec, cm_4_f1,  cm_5_acc, cm_5_recall, cm_5_prec, cm_5_f1,  cm_6_acc, cm_6_recall, cm_6_prec, cm_6_f1,  cm_7_acc, cm_7_recall, cm_7_prec, cm_7_f1,  cm_8_acc, cm_8_recall, cm_8_prec, cm_8_f1,  cm_9_acc, cm_9_recall, cm_9_prec, cm_9_f1,  cm_total_acc, cm_total_recall, cm_total_prec, cm_total_f1 = ConfusionMatrixCompute(ConfusionMatrix)
    print('Acurácia:', cm_total_acc, '\nRecall:', cm_total_recall, '\nPrecisão:', cm_total_prec, '\nF1_score:', cm_total_f1)

## Matriz de confusão
def ConfusionMatrixTree():
    A_resultado, clf = Tree()
    XaTrain, XaTest, yaTrain, yaTest = pre.Data()
    ConfusionMatrixTree = cm.ConfusionMatrix(yaTest,A_resultado)
    return ConfusionMatrixTree

def EvaluateTreeClouds(ConfusionMatrixTree):
    cm.ConfusionMatrixScoreClouds(ConfusionMatrixTree)

def EvaluateTreeTotal(ConfusionMatrixTree):
    cm.ConfusionMatrixScoreTotal(ConfusionMatrixTree)
