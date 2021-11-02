import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
from generateData import color
import itertools
def GMMem(K, iter, ClusterData):
    mus, sigmas, alpha = initParams(ClusterData, K)
    costAfter = 0
    gamma = 0
    for i in range(iter):
        cost = costAfter
        gamma = getExpectation(ClusterData,mus,sigmas,alpha,K)
        mus, sigmas, alpha = maximize(ClusterData,gamma,K)
        costAfter = calculateMLE(ClusterData,mus, sigmas, alpha ,K)
        if abs(costAfter - cost)<=1e-5:
            print("GMM EM 迭代次数为{}".format(i))
            print("cost = {}".format(cost))
            print("costAfter = {}".format(costAfter))
            print("GMM EM迭代完成")
            break
    return gamma,mus,sigmas,alpha


def calculateMLE(ClusterData,mus, sigmas, alpha, K):
    """
    计算似然函数
    :param ClusterData:所有数据点
    :param mus:多维高斯分布均值矩阵
    :param sigmas:多维高斯分布的协方差矩阵
    :param alpha:每个簇的权重
    :param K:簇的个数
    :return:MLE的值
    """
    iterN = 0
    for i in range(ClusterData.shape[0]):
        iterK = 0
        for k in range(K):
            iterK = iterK+alpha[k]*multivariate_normal.pdf(ClusterData[i], mus[k], sigmas[k])
        iterN = iterN+  math.log(iterK,math.e)
    return iterN


def getExpectation(ClusterData, mus, sigmas, alpha, K):
    """
    返回gamma矩阵
    :param ClusterData:所有数据点
    :param mus:多维高斯分布均值矩阵
    :param sigmas:多维高斯分布的协方差矩阵
    :param alpha:每个簇的权重
    :param K:簇的个数
    :return:gamma矩阵
    """
    gamma = np.zeros((ClusterData.shape[0], K))
    for i in range(ClusterData.shape[0]):
        gamma_sum = 0
        for j in range(K):
            gamma[i][j] = alpha[j] * multivariate_normal.pdf(ClusterData[i], mus[j], sigmas[j])
            gamma_sum = gamma_sum + gamma[i][j]
        for j in range(K):
            gamma[i][j] = gamma[i][j] / gamma_sum
    return gamma



def maximize(ClusterData, gamma, K):
    """
    M Step
    :param ClusterData:所有数据点
    :param gamma:gamma矩阵
    :param K:所有的类别
    :return:m step后的参数
    """
    mus = np.zeros((K,ClusterData.shape[1]))
    for i in range(K):
        iterNumerator = np.zeros(ClusterData.shape[1])
        iterDenominator = 0
        for j in range(ClusterData.shape[0]):
            iterNumerator = iterNumerator + gamma[j][i]*ClusterData[j]
            iterDenominator = iterDenominator+gamma[j][i]
        for k in range(ClusterData.shape[1]):
            # 存在除0可能
            mus[i][k] = iterNumerator[k] / iterDenominator

    sigmas = np.zeros((K, ClusterData.shape[1], ClusterData.shape[1]))
    for k in range(K):
        iterNumerator = np.zeros((ClusterData.shape[1],ClusterData.shape[1]))
        iterDenominator = 0
        for i in range(ClusterData.shape[0]):
            iterNumerator = iterNumerator + gamma[i][k] * np.dot(np.transpose(ClusterData[i] - mus[k]),(ClusterData[i] - mus[k]))
            iterDenominator = iterDenominator +gamma[i][k]
        sigma = iterNumerator / iterDenominator
        for j in range(ClusterData.shape[1]):
            sigma[j][j] = sigma[j][j]*1.01
        sigmas[k] = sigma

    alpha = np.zeros(K)
    for i in range(K):
        iterNumerator = 0
        for j in range(ClusterData.shape[0]):
            iterNumerator=iterNumerator+gamma[j][i]
        alpha[i] = iterNumerator/ClusterData.shape[0]
    return mus, sigmas, alpha


def initParams(ClusterData, K):
    """
    初始化GMM参数
    :param ClusterData:所有数据点
    :param K: 要分类的类别
    :return: GMM的各个参数
    """
    d = ClusterData.shape[1]
    # 高斯分布模型均值
    mus = np.random.rand(K, d)
    # 初始化协方差矩阵
    sigmas = np.array([np.eye(d)] * K)
    # 假设起始时各个类别概率相同
    alpha = np.ones(K) * (1 / K)
    return mus, sigmas, alpha



def drawGMM(ClusterData,gamma,mus):
    """
    画出GMM簇的图像
    :param ClusterData:所有数据点
    :param gamma: gamma矩阵
    """
    center = np.zeros(ClusterData.shape[0])
    for i in range(ClusterData.shape[0]):
        center[i] = gamma[i].argmax()
    plt.title("GMM EM algorithm")
    for i in range(gamma.shape[1]):
        listX1 = []
        listX2 = []
        for j in range(ClusterData.shape[0]):
            if int(center[j]) == i:
                listX1.append(ClusterData[j][0])
                listX2.append(ClusterData[j][1])
        plt.scatter(listX1, listX2,c=color[i])
    mu = np.transpose(mus)
    plt.scatter(mu[0],mu[1],c=color[gamma.shape[1]],s=100)
    plt.show()


def TestGMMAccuracy(centersTrain, mus,sigmas,alpha, YTrain, YTest,Y,Test, K):
    # 簇号
    Ytest = np.zeros(Test.shape[0])
    for i in range(Test.shape[0]):
        gamma= getExpectation(Test, mus, sigmas, alpha, K)
        Ytest[i] = np.array(gamma[i]).argmax()
    # Dict 簇到原来y的映射
    Dict = ClusterMap(K, YTrain,centersTrain,Y)
    sum = 0
    for i in range(YTest.shape[0]):
        Ytest[i] = Dict[Ytest[i]]
    for i in range(YTest.shape[0]):
        sum = sum +1
    Accuracy = sum/Ytest.shape[0]
    print("GMM EM iris数据集的正确率为{}".format(Accuracy))


def ClusterMap(K,YTrain, dataCenters,Y):
    Set = list(set(Y))
    List = []
    for i in range(K):
        List.append(i)
    listArray = list(itertools.permutations(List,K))
    MapAccuracyDict = np.zeros(len(listArray))
    listDict = []
    for j in range(len(listArray)):
        # 簇号
        Dict = {}
        for k in range(len(listArray[j])):
            Dict[listArray[j][k]] = Set[k]
        listDict.append(Dict)
        MapAccuracyDict[j] = MapAccuracy(Dict,YTrain,dataCenters)
    return listDict[np.array(MapAccuracyDict).argmax()]



def MapAccuracy(Dict,YTrain,dataCenters):
    sum = 0
    for i in range(YTrain.shape[0]):
        if Dict[dataCenters[i]] == YTrain[i]:
            sum = sum+1
    return sum/YTrain.shape[0]

