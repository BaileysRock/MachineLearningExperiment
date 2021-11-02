import numpy as np
import matplotlib.pyplot as plt
import math
from generateData import color
import itertools

def initCenters(KMeansData, K):
    """
    初始化中心点
    :param KMeansData:所有数据点
    :param K: 分成类别的个数
    :return:
    """
    centerX1 = np.random.choice(KMeansData[0], K)
    centerX2 = np.random.choice(KMeansData[1], K)
    center = np.vstack((centerX1, centerX2))
    return center


def kMeansCost(KMeansData, centers, DataCenter):
    """
    计算当前所有KMeansData中的所有点与centers所有点的代价函数的值
    :param KMeansData: 所有KMeansData的数据点
    :param centers: 中心点
    :param DataCenter: 每个点对应的中心点
    :return:
    """
    cost = 0
    for i in range(KMeansData.shape[1]):
        X1 = KMeansData[0][i]
        X2 = KMeansData[1][i]
        centerX1 = centers[0][int(DataCenter[i])]
        centerX2 = centers[1][int(DataCenter[i])]
        cost = cost + math.sqrt((X1 - centerX1) * (X1 - centerX1) + (X2 - centerX2) * (X2 - centerX2))
    return cost


def calculateDistanceCenter(X1, X2, center):
    """
    计算当前的点与中心点的距离，选择最近的类
    :param X1: 点的坐标
    :param X2: 点的坐标
    :param center: 中心点数组
    :return: 点的类
    """
    cost = float("inf")
    minCost = cost
    Xclass = 0
    for i in range(center.shape[1]):
        cost = 0
        cost = cost + math.sqrt((X1 - center[0][i]) * (X1 - center[0][i]))
        cost = cost + math.sqrt((X2 - center[1][i]) * (X2 - center[1][i]))
        if cost < minCost:
            minCost = cost
            Xclass = i
    return Xclass


def assignClass(KMeansData, Centers):
    """
    为KMeansData每个数据分配中信
    :param KMeansData: 数据
    :param Centers: 中心点
    :return: 为每个点分配离它最近的中心点的序号
    """
    center = np.zeros(KMeansData.shape[1])
    for i in range(KMeansData.shape[1]):
        center[i] = calculateDistanceCenter(KMeansData[0][i], KMeansData[1][i], Centers)
    return center


def updateCenters(KMeansData, centers, K):
    """
    更新中心点K个中心点
    :param KMeansData:数据
    :param centers:每个数据的中心点
    :param K:类别个数
    :return:更新新的中心点
    """
    UpdateCenters = np.zeros([2, K])
    list1 = []
    list2 = []
    for i in range(K):
        for j in range(KMeansData.shape[1]):
            if centers[j] == i:
                list1.append(KMeansData[0][j])
                list2.append(KMeansData[1][j])
        x1 = 0
        x2 = 0
        for k in range(len(list1)):
            x1 = x1 + list1[k]
            x2 = x2 + list2[k]

        if x1 == 0 and x2 == 0:
            print("此时无点离他最近")
            UpdateCenters[0][i] = 0
            UpdateCenters[1][i] = 0
            # randomNum = random.randint(0, KMeansData.shape[1]-1)
            # UpdateCenters[0][i] = KMeansData[0][randomNum]
            # UpdateCenters[1][i] = KMeansData[1][randomNum]
        else:
            x1 = x1 / len(list1)
            x2 = x2 / len(list2)
            UpdateCenters[0][i] = x1
            UpdateCenters[1][i] = x2
        list1 = []
        list2 = []
    return UpdateCenters


def kMeans(K, iter, ClusterData):
    """
    调用Kmeans算法
    :param K: 类别个数
    :param iter: 迭代次数
    :param ClusterData: 实验数据
    :return:
    """
    # 初始化开始时候的K个中心点
    centers = initCenters(ClusterData, K)
    costAfter = 0
    DataCenter = np.zeros(ClusterData.shape[1])
    for i in range(iter):
        cost = costAfter
        # 为每个点分配中心
        DataCenter = assignClass(ClusterData, centers)
        # 更新中心点
        centers = updateCenters(ClusterData, DataCenter, K)
        costAfter = kMeansCost(ClusterData, centers, DataCenter)
        if (costAfter == cost):
            print("K-Means 迭代次数为{}".format(i))
            print("cost = {}".format(cost))
            print("costAfter = {}".format(costAfter))
            print("K-Means迭代完成")
            # drawKmeansClassify(ClusterData, DataCenter, centers,K)
            break
    return DataCenter, centers


def drawKmeansClassify(KMeansData, DataCenter, centers, K):
    """
    为K-Means算法生成的结果画图
    :param KMeansData:KMeans的数据
    :param DataCenter:每个点对应的中心
    :param K:类别的个数
    """
    plt.title("K-Means algorithm")
    for i in range(K):
        listX1 = []
        listX2 = []
        for j in range(KMeansData.shape[1]):
            if DataCenter[j] == i:
                listX1.append(KMeansData[0][j])
                listX2.append(KMeansData[1][j])
        plt.scatter(listX1, listX2, c=color[i])
    plt.scatter(centers[0], centers[1], c=color[K], s=100)
    plt.xlabel("$X_{1}$")
    plt.ylabel("$X_{2}$")
    plt.show()


def TestKMeansAccuracy(centersTrain, centers, YTrain, YTest,Y,KMeansTest, K):

    # 簇号
    Ytest = np.zeros(KMeansTest.shape[1])
    for i in range(KMeansTest.shape[1]):
        Ytest[i] = calculateDistanceCenter(KMeansTest[0][i], KMeansTest[1][i], centers)
    # Dict 簇到原来y的映射
    Dict = KMeansClusterMap(K, YTrain,centersTrain,Y)
    sum = 0
    for i in range(YTest.shape[0]):
        Ytest[i] = Dict[Ytest[i]]
    for i in range(YTest.shape[0]):
        sum = sum +1
    Accuracy = sum/Ytest.shape[0]
    print("K-means iris数据集的正确率为{}".format(Accuracy))


def KMeansClusterMap(K,YTrain, dataCenters,Y):
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
        MapAccuracyDict[j] = KMeansMapAccuracy(Dict,YTrain,dataCenters)
    return listDict[np.array(MapAccuracyDict).argmax()]



def KMeansMapAccuracy(Dict,YTrain,dataCenters):
    sum = 0
    for i in range(YTrain.shape[0]):
        if Dict[dataCenters[i]] == YTrain[i]:
            sum = sum+1
    return sum/YTrain.shape[0]





