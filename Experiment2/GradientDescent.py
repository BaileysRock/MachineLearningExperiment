import numpy as np
from ComputeCost import computeCost


def gradientDescent(X, Y, alpha, Lambda, precision, iterNum):
    """
    使用梯度下降法进行迭代
    :param X: 数据特征
    :param Y: 数据分类
    :param alpha: 学习率
    :param Lambda: 惩罚项系数
    :param precision: 迭代精度
    :param iterNum: 迭代次数
    :return:
    """
    theta = np.zeros(X.shape[1])
    descentStore = np.zeros(iterNum + 1)
    iterStore = np.zeros(iterNum + 1)
    iterStore[0] = np.inf
    for i in range(1, iterNum):
        iter = 0
        for j in range(0, X.shape[0]):

            # WX = np.matmul(theta, np.reshape(X[j], (-1, 1)))
            # ExpWX = np.exp(WX)[0]
            # iter = iter + (-X[j] * Y[j] + (ExpWX/ (1 + ExpWX)) * X[j])

            WX = np.matmul(theta, np.reshape(X[j], (-1, 1)))
            ExpWX = np.exp(WX)[0]

            if ExpWX == np.inf:
                iter = iter + (-X[j] * Y[j] + X[j])
            else:
                iter = iter + (-X[j] * Y[j] + (ExpWX / (1 + ExpWX)) * X[j])

        # if i % 1000 == 0:
        #     print("np.sum(iter)")
        #     print(abs(np.sum(iter)))
        iterStore[i + 1] = np.sum(iter)
        # if abs(iterStore[i+1]) > abs(iterStore[i]):
        #     alpha = alpha/2
        #     print("当前alpha={}".format(alpha))
        # print("iterStore[i+1]")
        # print(iterStore[i+1])
        # print("iterStore[i+1]-iterStore[i]")
        # print(abs(iterStore[i+1]-iterStore[i]))
        # if abs(iterStore[i+1]-iterStore[i]) <= precision:
        #     # print(i)
        #     print("迭代次数为{}".format(i - 1))
        #     print(abs(descentStore[i]-descentStore[i-1]))
        #     break
        iter = iter + Lambda * theta
        theta = theta - alpha * iter
        descentStore[i] = computeCost(X, Y, theta, Lambda)
        # if abs(descentStore[i]-descentStore[i-1])<=precision:
        #     print(i)
        #     print("迭代次数为{}".format(i-1))
        #     # print("descentStore[i]-descentStore[i-1]")
        #     # print(abs(descentStore[i]-descentStore[i-1]))
        #     break
    return theta
