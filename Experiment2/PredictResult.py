from GradientDescent import gradientDescent
from GenerateData import TrainTestSplit
from DataFromFile import readFromFile
from VerificationPredict import predictPrecision
from NewtonMethod import newtonMethod
from GenerateData import generateData
from Draw import drawTheta
from Draw import plotScatter
import numpy as np
import matplotlib.pyplot as plt


def predictResult(method, FileName=""):
    """
    从文件中读取数据并根据不同的method使用不同方法求得极小值
    并通过调用自己写的精度函数验证预测准确率
    :param method: 使用的方法，0:对于符合朴素贝叶斯分布的数据 1：对于不符合朴素贝叶斯分布的数据 2：梯度下降法 3：牛顿迭代法
    :param FileName: 文件名称
    """

    if method == 0:
        X, Y, X1Y0, X2Y0, X1Y1, X2Y1 = generateData(0)
        X_train, X_test, Y_train, Y_test = TrainTestSplit(X, Y)
        theta = gradientDescent(X_train, Y_train, 0.01, 1e-6, 1e-7, 20000)
        plotScatter(X1Y0, X2Y0, X1Y1, X2Y1)
        drawTheta(-1, 5, theta, 0)
        predictPrecision(theta, X_test, Y_test)
        theta = gradientDescent(X_train, Y_train, 0.01, 0, 1e-7, 20000)
        plotScatter(X1Y0, X2Y0, X1Y1, X2Y1)
        drawTheta(-1, 5, theta, 0)
        predictPrecision(theta, X_test, Y_test)
        theta = newtonMethod(X_train, Y_train, 1e-6, 1e-7, 20000)
        plotScatter(X1Y0, X2Y0, X1Y1, X2Y1)
        drawTheta(-1, 5, theta, 1)
        predictPrecision(theta, X_test, Y_test)
        theta = newtonMethod(X_train, Y_train, 0, 1e-7, 20000)
        plotScatter(X1Y0, X2Y0, X1Y1, X2Y1)
        drawTheta(-1, 5, theta, 1)
        predictPrecision(theta, X_test, Y_test)

    elif method == 1:
        X, Y, X1Y0, X2Y0, X1Y1, X2Y1 = generateData(1)
        X_train, X_test, Y_train, Y_test = TrainTestSplit(X, Y)
        theta = gradientDescent(X_train, Y_train, 0.001, 1e-7, 1e-7, 20000)
        plotScatter(X1Y0, X2Y0, X1Y1, X2Y1)
        drawTheta(-1, 5, theta, 0)
        predictPrecision(theta, X_test, Y_test)
        theta = gradientDescent(X_train, Y_train, 0.001, 0, 1e-7, 20000)
        plotScatter(X1Y0, X2Y0, X1Y1, X2Y1)
        drawTheta(-1, 5, theta, 0)
        predictPrecision(theta, X_test, Y_test)
        theta = newtonMethod(X_train, Y_train, 1e-7, 1e-7, 20000)
        plotScatter(X1Y0, X2Y0, X1Y1, X2Y1)
        drawTheta(-1, 5, theta, 1)
        predictPrecision(theta, X_test, Y_test)
        theta = newtonMethod(X_train, Y_train, 0, 1e-7, 20000)
        plotScatter(X1Y0, X2Y0, X1Y1, X2Y1)
        drawTheta(-1, 5, theta, 1)
        predictPrecision(theta, X_test, Y_test)


    elif method == 2:
        X, Y = readFromFile(FileName)
        X_train, X_test, Y_train, Y_test = TrainTestSplit(X, Y)
        theta = gradientDescent(X_train, Y_train, 0.0001, 1e-7, 1e-8, 50000)
        print(FileName)
        print("梯度下降法测试结果测试结果：")
        predictPrecision(theta, X_test, Y_test)
    elif method == 3:
        X, Y = readFromFile(FileName)
        X_train, X_test, Y_train, Y_test = TrainTestSplit(X, Y)
        theta = newtonMethod(X_train, Y_train, 1e-7, 1e-8, 50000)
        print(FileName)
        print("牛顿迭代法测试结果测试结果：")
        predictPrecision(theta, X_test, Y_test)
