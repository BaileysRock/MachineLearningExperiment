import numpy as np

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
