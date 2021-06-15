import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

## Função faz a leitura dos dados e aplica o método chunks para ler cada parte dos dados separadamente e depois juntá-los novamente para otimizar a leitura
def loadData():
    chunks = []
    for chunk in pd.read_csv('C:/Users/Ygoor/Desktop/Freelas/FreeCloud/NewCloud/dados.csv', encoding = 'UTF-8', low_memory = True, chunksize=1000):
        chunks.append(chunk)
    df = pd.concat(chunks)
    return df

## Função exclusivamente para fazer a limpeza dos dados e reorganizar eles para o melhor jeito em que possamos usá-los
def CleanData():
    df = loadData()

## Exclusão das colunas que não serão usadas
    df = df.drop(['Estação'], axis = 1)

## Conversão do formato da data
    df['Data'] = pd.to_datetime(df['Data'])
    df['Data'] = df['Data'].dt.strftime('%Y-%m-%d')

## Conversão de horário
    df['Horario'] = pd.to_datetime(df['Horario'], format = '%H%M')
    df['Horario'] = df['Horario'].dt.strftime('%H:%M:%S')

## Reorganização das colunas
    df = df[['Nuvem Baixa','Nuvem Média', 'Nuvem Alta', 'Data', 'Horario', 'Banda 1', 'Banda 1 Media', 'Banda 1 Desv. Pad', 'Banda 2', 'Banda 2 Media', 'Banda 2 Desvio Padrão', 'Banda 3', 'Banda 3 Media', 'Banda 3 Desvio Padrão', 'Banda 4', 'Banda 4 Media', 'Banda 4 Desvio Padrão', 'Banda 5', 'Banda 5 Media', 'Banda 5 Desvio Padrão', 'Banda 6', 'Banda 6 Media', 'Banda 6 Desvio Padrão', 'Banda 7', 'Banda 7 Media', 'Banda 7 Desvio Padrão', 'Banda 8', 'Banda 8 Media', 'Banda 8 Desvio Padrão', 'Banda 9', 'Banda 9 Media', 'Banda 9 Desvio Padrão', 'Banda 10', 'Banda 10 Media', 'Banda 10 Desvio Padrão', 'Banda 11', 'Banda 11 Media', 'Banda 11 Desvio Padrão', 'Banda 12', 'Banda 12 Media', 'Banda 12 Desvio Padrão', 'Banda 13', 'Banda 13 Media', 'Banda 13 Desvio Padrão', 'Banda 14', 'Banda 14 Media', 'Banda 14 Desvio Padrão', 'Banda 15', 'Banda 15 Media', 'Banda 15 Desvio Padrão', 'Banda 16', 'Banda 16 Media', 'Banda 16 Desvio Padrão', 'Latitude', 'Longitude']]
    return df

## Separação dos alvos de acordo com suas características
def SepareData():
    df = CleanData()
    print("Qual tipo de nuvem?")
    print("...................")
    db = input()
    if db == "alta":
        alta = df[(df.iloc[:,0] >= 0) & (df.iloc[:,1] >= 0) & (df.iloc[:,2] > 0)]
        Ya = alta.iloc[:,2]
        Xa = alta.iloc[:,3:]
    return Xa, Ya

    if db == "baixa":
        baixa = df[(df.iloc[:,0] > 0) & (df.iloc[:,1] == 0) & (df.iloc[:,2] == 0)]
        Ya = baixa.iloc[:,0]
        Xa = baixa.iloc[:,3:]
    return Xa, Ya

    if db == "media":
        media = df[(df.iloc[:,0] >= 0) & (df.iloc[:,1] > 0) & (df.iloc[:,2] == 0)]
        Ya = media.iloc[:,1]
        Xa = media.iloc[:,3:]
    return Xa, Ya

    if db == "superficie":
        superficie = df[(df.iloc[:,0] == 0) & (df.iloc[:,1] == 0) & (df.iloc[:,2] == 0)]
        Ya = superficie.iloc[:,2]
        Xa = superficie.iloc[:,3:]
    return Xa, Ya


def Categorization(X):
    X = pd.get_dummies(X, columns = ['Data', 'Horario'], sparse = False)
    return X

## Aplica a técnica SMOTE para criar novos dados que são minoritários no conjunto de dados
def SMOTE_apply(X, y):
  smote = SMOTE()
  X_smote, y_smote = smote.fit_resample(X, y)
  return X_smote, y_smote

## Função para aplicar o método PCA para reduzir a dimensionalidade dos dados
def PCA_apply(X):
  pca = PCA()
  X.iloc[:,0:49] = pca.fit_transform(X.iloc[:,0:49])
  X.iloc[:,0:49] = pd.DataFrame(X.iloc[:,0:49])
  return X

## Função para aplicar o método SMOTE para os dados do conjunto
def SmotePCAData():
    Xa, Ya = CleanData()
    Xa = Categorization(Xa)
    Xa_balanced , Ya_balanced = SMOTE_apply(Xa, Ya)
    Xa_balanced = PCA_apply(Xa_balanced)
    return Xa_balanced, Ya_balanced

## Separa os dados em dados de treino e teste
def TrainTestAlta():
    Xa_balanced , Ya_balanced = SmotePCAData()
    XaTrain, XaTest, yaTrain, yaTest = train_test_split(Xa_balanced, Ya_balanced, test_size = 0.2)
    return XaTrain, XaTest, yaTrain, yaTest

def ScallingData(XTrain, XTest):
    scaleX = StandardScaler()
    XTrain.iloc[:,0:49] = scaleX.fit_transform(XTrain.iloc[:,0:49])
    XTest.iloc[:,0:49] = scaleX.fit_transform(XTest.iloc[:,0:49])
    return XTrain, XTest

def Data():
    XaTrain, XaTest, yaTrain, yaTest = TrainTestAlta()
    XaTrain, XaTest = ScallingData(XaTrain, XaTest)
    return XaTrain, XaTest, yaTrain, yaTest

## Função para separar somente os dados utilizados para análise do primeiro canal
def Chnl(XTrain, XTest):
    print("Banda: ")
    Canal = int(input())
    Media = Canal + 1
    Desvio = Media + 1
    XTrain = XTrain.iloc[:,[Canal, Media, Desvio, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 58, 60, 71, 72, 73, 74, 75, 76, 77, 78, 78, 80]]
    XTest = XTest.iloc[:,[Canal, Media, Desvio, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 58, 60, 71, 72, 73, 74, 75, 76, 77, 78, 78, 80]]
    return XTrain, XTest
