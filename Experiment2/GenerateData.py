import numpy as np
from sklearn.model_selection import train_test_split


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
    :param Class: 类别
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


def generateData(method):
    """
    生成两类数据并满足朴素贝叶斯分布
    :param method: 分别生成符合朴素贝叶斯分布和不符合朴素贝叶斯分布的数据
    :return: 生成数据
    """
    if method == 1:
        # Y0Mu1 = 2
        # Y0Mu2 = 2
        # Y0cov11 = 1
        # Y0cov12 = -0.5
        # Y0cov21 = -0.5
        # Y0cov22 = 1
        # Y0Num = 200
        # Y0noiseSigma1 = 0.2
        # Y0noiseSigma2 = 0.4
        # X1Y, X2Y = generate2DimensionalData(Y0Mu1, Y0Mu2, Y0cov11, Y0cov12, Y0cov21, Y0cov22, Y0Num,
        #                                       Y0noiseSigma1, Y0noiseSigma2)
        # X1Y0 = X1Y[0:np.int64(Y0Num/2)]
        # X1Y1 = X1Y[np.int64(Y0Num/2):np.int64(Y0Num)]
        # X2Y0 = X2Y[0:np.int64(Y0Num/2)]
        # X2Y1 = X2Y[np.int64(Y0Num/2):np.int64(Y0Num)]
        # XY0 = np.dstack((X1Y0, X2Y0))
        # XY1 = np.dstack((X1Y1, X2Y1))
        # XY0 = addUnitColumn(XY0)
        # XY1 = addUnitColumn(XY1)
        # X = np.vstack((XY0, XY1))
        # Y_zero = np.zeros([XY0.shape[0]])
        # Y_ones = np.ones([XY1.shape[0]])
        # Y_zero = np.reshape(Y_zero, (-1, 1))
        # Y_ones = np.reshape(Y_ones, (-1, 1))
        # Y = np.vstack((Y_zero, Y_ones))
        # Y = np.reshape(Y, (-1))
        # return X, Y, X1Y0, X2Y0, X1Y1, X2Y1

        # Y0类生成数据的参数
        Y0Mu1 = 2
        Y0Mu2 = 2
        Y0cov11 = 1
        Y0cov12 = -0.5
        Y0cov21 = -0.5
        Y0cov22 = 0.7
        Y0Num = 100
        Y0noiseSigma1 = 0.2
        Y0noiseSigma2 = 0.4
        # Y1类生成数据的参数
        Y1Mu1 = 1
        Y1Mu2 = -1
        Y1cov11 = 1
        Y1cov12 = -0.6
        Y1cov21 = -0.6
        Y1cov22 = 1.3
        Y1Num = 100
        Y1noiseSigma1 = 0.1
        Y1noiseSigma2 = 0.2
        X1Y0, X2Y0 = generate2DimensionalData(Y0Mu1, Y0Mu2, Y0cov11, Y0cov12, Y0cov21, Y0cov22, Y0Num,
                                              Y0noiseSigma1, Y0noiseSigma2)
        X1Y1, X2Y1 = generate2DimensionalData(Y1Mu1, Y1Mu2, Y1cov11, Y1cov12, Y1cov21, Y1cov22, Y1Num,
                                              Y1noiseSigma1, Y1noiseSigma2)
        XY0 = np.dstack((X1Y0, X2Y0))
        XY1 = np.dstack((X1Y1, X2Y1))
        XY0 = addUnitColumn(XY0)
        XY1 = addUnitColumn(XY1)
        X = np.vstack((XY0, XY1))
        Y_zero = np.zeros([XY0.shape[0]])
        Y_ones = np.ones([XY1.shape[0]])
        Y_zero = np.reshape(Y_zero, (-1, 1))
        Y_ones = np.reshape(Y_ones, (-1, 1))
        Y = np.vstack((Y_zero, Y_ones))
        Y = np.reshape(Y, (-1))
        return X, Y, X1Y0, X2Y0, X1Y1, X2Y1



    elif method == 0:
        # Y0类生成数据的参数
        Y0Mu1 = 2
        Y0Mu2 = 2
        Y0cov11 = 0.5
        Y0cov12 = 0
        Y0cov21 = 0
        Y0cov22 = 1
        Y0Num = 100
        Y0noiseSigma1 = 0.2
        Y0noiseSigma2 = 0.4
        # Y1类生成数据的参数
        Y1Mu1 = 1
        Y1Mu2 = -1
        Y1cov11 = 0.5
        Y1cov12 = 0
        Y1cov21 = 0
        Y1cov22 = 1
        Y1Num = 100
        Y1noiseSigma1 = 0.1
        Y1noiseSigma2 = 0.2
        X1Y0, X2Y0 = generate2DimensionalData(Y0Mu1, Y0Mu2, Y0cov11, Y0cov12, Y0cov21, Y0cov22, Y0Num,
                                              Y0noiseSigma1, Y0noiseSigma2)
        X1Y1, X2Y1 = generate2DimensionalData(Y1Mu1, Y1Mu2, Y1cov11, Y1cov12, Y1cov21, Y1cov22, Y1Num,
                                              Y1noiseSigma1, Y1noiseSigma2)
        XY0 = np.dstack((X1Y0, X2Y0))
        XY1 = np.dstack((X1Y1, X2Y1))
        XY0 = addUnitColumn(XY0)
        XY1 = addUnitColumn(XY1)
        X = np.vstack((XY0, XY1))
        Y_zero = np.zeros([XY0.shape[0]])
        Y_ones = np.ones([XY1.shape[0]])
        Y_zero = np.reshape(Y_zero, (-1, 1))
        Y_ones = np.reshape(Y_ones, (-1, 1))
        Y = np.vstack((Y_zero, Y_ones))
        Y = np.reshape(Y, (-1))
        return X, Y, X1Y0, X2Y0, X1Y1, X2Y1


def addUnitColumn(X):
    """
    为x添加全为1的列
    :param X: 待添加的矩阵
    :return: 添加后的矩阵
    """
    UnitMatrix = np.ones(X.shape[1])
    # UnitMatrix = np.reshape(UnitMatrix,(X.shape[0],-1))
    X = np.dstack((UnitMatrix, X))
    X = np.reshape(X, (X.shape[1], X.shape[2]))
    return X


def TrainTestSplit(X, Y):
    """
    将数据集分为训练集和测试集
    :param X: 训练特征数据
    :param Y: 训练数据对应分类
    :return: 分开后的数据集
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)
    return X_train, X_test, Y_train, Y_test
