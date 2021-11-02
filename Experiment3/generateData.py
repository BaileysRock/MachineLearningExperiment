import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

color = ['c', 'r', 'b', 'm', 'y', 'k', 'g', 'w']
muList = [[5, 0], [3, 3], [1, 5], [4, 2], [1, 3], [2, 3]]


def generate2DimensionalData(Mu1, Mu2, cov11, cov12, cov21, cov22, Num, noiseSigma1, noiseSigma2):
    """
    生成二维高斯分布数据
    :param Mu1: 维度1的平均值
    :param Mu2: 维度2的平均值
    :param cov11: 维度1的方差
    :param cov12: 维度12的协方差
    :param cov21: 维度21的协方差
    :param cov22: 维度2的方差
    :param Num: 生成样本点的个数
    :param noiseSigma1: 维度1噪声的方差
    :param noiseSigma2: 维度2噪声的方差
    :return: 分别返回两个维度的样本点
    """
    mean = np.array([Mu1, Mu2])
    cov = np.array([[cov11, cov12], [cov21, cov22]])
    Data = np.random.multivariate_normal(mean, cov, Num)
    X1 = Data[:, 0]
    X2 = Data[:, 1]
    # 添加高斯噪声
    GuassNoise1 = np.random.normal(0, scale=noiseSigma1, size=Num)
    GuassNoise2 = np.random.normal(0, scale=noiseSigma2, size=Num)
    X1 = X1 + GuassNoise1
    X2 = X2 + GuassNoise2
    return X1, X2


def generateData(K):
    """
    生成K堆二维高斯分布数据
    :param K: 生成样本点的堆数
    :return: X1、X2的列表
    """
    X1 = np.empty(0)
    X2 = np.empty(0)
    plt.title("Origin Data")
    number = 50
    # Y = np.empty(0)
    for i in range(K):
        # X1gen, X2gen = generate2DimensionalData(random.randint(0, 5), random.randint(0, 5), 1, 0, 0, 1, number, 0.1, 0.1)
        X1gen, X2gen = generate2DimensionalData(muList[i][0], muList[i][1], 1, 0, 0, 1, number, 0.1, 0.1)
        # Y = np.hstack((Y,np.ones(number)*i))
        X1 = np.hstack((X1, X1gen))
        X2 = np.hstack((X2, X2gen))
        plt.scatter(X1gen, X2gen, c=color[i])
    plt.show()
    return X1, X2


def makeDataForKmeans(X1, X2):
    """
    将数据处理，适合KMeans算法
    :param X1: 一行数据
    :param X2: 一行数据
    :return: 两行数据
    """
    X = np.vstack((X1, X2))
    return X


def KmeansTrainTestSplit(ClusterData, Y):
    clusterData = np.transpose(ClusterData)
    X_train, X_test, Y_train, Y_test = train_test_split(clusterData, Y)
    X_train = np.transpose(X_train)
    X_test = np.transpose(X_test)
    return X_train, X_test, Y_train, Y_test


def makeDataForGMM(X1, X2):
    X1 = np.reshape(X1, (-1, 1))
    X2 = np.reshape(X2, (-1, 1))
    X = np.column_stack((X1, X2))
    return X
