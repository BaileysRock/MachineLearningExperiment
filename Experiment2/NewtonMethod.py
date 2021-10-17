import numpy as np


def newtonMethod(X, Y, Lambda, precision, iterNum):
    """
    使用牛顿迭代法进行迭代
    :param X: 数据特征
    :param Y: 数据分类
    :param Lambda: 惩罚项系数
    :param precision: 预测精度
    :param iterNum: 迭代次数
    :return: theta
    """
    theta = np.zeros(X.shape[1])
    iterStore = np.zeros(iterNum + 1)
    iterStore[0] = np.inf
    for i in range(0, iterNum):
        iterComputeDifferential = 0
        iterCompute = 0
        for j in range(0, X.shape[0]):
            ExpWX = np.exp(np.dot(theta, np.reshape(X[j], (-1, 1))))
            iterComputeDifferential = iterComputeDifferential + (ExpWX * np.dot(X[j], np.reshape(X[j], (-1, 1)))) / (
                        1 + ExpWX)

            ExpWX = np.exp(np.matmul(theta, np.reshape(X[j], (-1, 1))))[0]
            iterCompute = iterCompute + (-X[j] * Y[j] + (ExpWX / (1 + ExpWX)) * X[j])

        iterStore[i + 1] = np.sum(iterCompute)
        # if abs(iterStore[i+1]-iterStore[i]) <= precision:
        #     print(i)
        #     print("迭代次数为{}".format(i - 1))
        #     break

        if abs(iterStore[i + 1] - iterStore[i]) <= precision:
            print(i)
            print("迭代次数为{}".format(i - 1))
            break

        iterCompute = iterCompute + Lambda * theta
        iterComputeDifferential = iterComputeDifferential + Lambda
        theta = theta - iterCompute / iterComputeDifferential
    return theta
