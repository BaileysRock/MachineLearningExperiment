import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def readFromFile(DocumentName):
    """
    从文件种读取数据
    :param DocumentName: 文件的名字
    :return: 数据
    """
    df = pd.read_csv(DocumentName)
    df.rename(columns={df.columns.array[df.columns.shape[0] - 1]: 'Predict'}, inplace=True)
    df['Predict'] = LabelEncoder().fit_transform(df['Predict'])
    Y = df['Predict']
    X = df.drop('Predict', axis=1)
    X = np.array(X.values)
    Y = np.array(Y.values)
    K = len(set(Y))
    return X, Y, K


def makeFileDataForKmeans(X, Y):
    """
    使数据适合写好的模型
    :param X: 数据点
    :return: 修改后的数据
    """
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y)
    Xtrain = np.transpose(Xtrain)
    Xtest = np.transpose(Xtest)
    Ytrain = np.transpose(Ytrain)
    Ytest = np.transpose(Ytest)
    return Xtrain, Xtest, Ytrain, Ytest


def makeFileDataForGMM(X, Y):
    """
    使数据适合写好的模型
    :param X: 数据点
    :return: 修改后的数据
    """
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y)
    return Xtrain, Xtest, Ytrain, Ytest
