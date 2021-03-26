import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def loadData():
    chunks = []
    for chunk in pd.read_csv('C:/Users/Ygoor/Desktop/Freelas/Projeto FreeCloud/dados.csv', encoding = 'UTF-8', low_memory = True, chunksize=1000):
        chunks.append(chunk)
    df = pd.concat(chunks)
    return df

def CleanConverseData():
    df = loadData()
## Exclusão das colunas que não serão usadas
    df = df.drop(['Estação'], axis = 1)
## Conversão do formato da data
    df['Data'] = pd.to_datetime(df['Data'])
    df['Data'] = df['Data'].dt.strftime('%Y-%m-%d')
## Conversão de horário
    df['Horario'] = pd.to_datetime(df['Horario'], format = '%H%M')
    df['Horario'] = df['Horario'].dt.strftime('%H:%M:%S')
## Junção das das duas colunas com formato datime para uma única
    df['Time'] = df['Data'] + " " + df['Horario']
## Exclusão das duas colunas anteriores
    df.drop(['Data','Horario'], axis = 1, inplace = True)
## Reorganização das colunas
    df = df[['Time', 'Nuvem Baixa','Nuvem Média', 'Nuvem Alta', 'Latitude', 'Longitude', 'Banda 1', 'Banda 2', 'Banda 3', 'Banda 4', 'Banda 5', 'Banda 6', 'Banda 7', 'Banda 8', 'Banda 9', 'Banda 10', 'Banda 11', 'Banda 12', 'Banda 13', 'Banda 14', 'Banda 15', 'Banda 16', 'Banda 1 Media', 'Banda 2 Media', 'Banda 3 Media', 'Banda 4 Media', 'Banda 5 Media', 'Banda 6 Media', 'Banda 7 Media', 'Banda 8 Media', 'Banda 9 Media', 'Banda 10 Media', 'Banda 11 Media', 'Banda 12 Media', 'Banda 13 Media', 'Banda 14 Media', 'Banda 15 Media', 'Banda 16 Media', 'Banda 1 Desv. Pad', 'Banda 2 Desvio Padrão', 'Banda 3 Desvio Padrão', 'Banda 4 Desvio Padrão', 'Banda 5 Desvio Padrão', 'Banda 6 Desvio Padrão', 'Banda 7 Desvio Padrão', 'Banda 8 Desvio Padrão', 'Banda 9 Desvio Padrão', 'Banda 10 Desvio Padrão', 'Banda 11 Desvio Padrão', 'Banda 12 Desvio Padrão', 'Banda 13 Desvio Padrão', 'Banda 14 Desvio Padrão', 'Banda 15 Desvio Padrão', 'Banda 16 Desvio Padrão', 'Banda 1 Variância', 'Banda 2 Variância', 'Banda 3 Variância', 'Banda 4 Variância', 'Banda 5 Variância', 'Banda 6 Variância', 'Banda 7 Variância', 'Banda 8 Variância', 'Banda 9 Variância', 'Banda 10 Variância', 'Banda 11 Variância', 'Banda 12 Variância', 'Banda 13 Variância', 'Banda 14 Variância', 'Banda 15 Variância', 'Banda 16 Variância']]
## Separação dos alvos de acordo com suas características
    superficie = df[(df.iloc[:,1] == 0) & (df.iloc[:,2] == 0) & (df.iloc[:,3] == 0)]
    baixa = df[(df.iloc[:,1] > 0) & (df.iloc[:,2] == 0) & (df.iloc[:,3] == 0)]
    media = df[(df.iloc[:,1] >= 0) & (df.iloc[:,2] > 0) & (df.iloc[:,3] == 0)]
    alta = df[(df.iloc[:,1] >= 0) & (df.iloc[:,2] >= 0) & (df.iloc[:,3] > 0)]
    return superficie, baixa, media, alta, df

def SepareScallingData():
    superficie, baixa, media, alta, df = CleanConverseData()
## Separação das variáveis alvo
    Ys = superficie.iloc[:,3].values
    Yb = baixa.iloc[:,1].values
    Ym = media.iloc[:,2].values
    Ya = alta.iloc[:,3].values
## Separação do espaço amostral para cada alvo
    Xs = superficie.iloc[:,3:21].values
    Xb = baixa.iloc[:,3:21].values
    Xm = media.iloc[:,3:21].values
    Xa = alta.iloc[:,3:21].values
## Chamada da função normalizadora
    scaleX = StandardScaler()
    Xs = scaleX.fit_transform(Xs.astype(float))
    Xb = scaleX.fit_transform(Xb.astype(float))
    Xm = scaleX.fit_transform(Xm.astype(float))
    Xa = scaleX.fit_transform(Xa.astype(float))
    return Xs, Ys, Xb, Yb, Xm, Ym, Xa, Ya

def TrainTestSuperficie():
    Xs, Ys, Xb, Yb, Xm, Ym, Xa, Ya = SepareScallingData()
    XsTrain, XsTest, ysTrain, ysTest = train_test_split(Xs, Ys, test_size = 0.2)
    return XsTrain, XsTest, ysTrain, ysTest

def TrainTestBaixa():
    Xs, Ys, Xb, Yb, Xm, Ym, Xa, Ya = SepareScallingData()
    XbTrain, XbTest, ybTrain, ybTest = train_test_split(Xb, Yb, test_size = 0.2)
    return XbTrain, XbTest, ybTrain, ybTest

def TrainTestMedia():
    Xs, Ys, Xb, Yb, Xm, Ym, Xa, Ya = SepareScallingData()
    XmTrain, XmTest, ymTrain, ymTest = train_test_split(Xm, Ym, test_size = 0.2)
    return XmTrain, XmTest, ymTrain, ymTest

def TrainTestAlta():
    Xs, Ys, Xb, Yb, Xm, Ym, Xa, Ya = SepareScallingData()
    XaTrain, XaTest, yaTrain, yaTest = train_test_split(Xa, Ya, test_size = 0.2)
    return XaTrain, XaTest, yaTrain, yaTest
