import numpy as np
import math
import matplotlib.pyplot as plt

from computeCost import ComputeCost
def CalculateLambda(X, y, theta,Lambda):
    hypthesis = np.matmul(X, theta)
    y = np.reshape(y,(X.shape[0],-1))
    # 先转置再做矩阵乘法
    cost = np.dot(np.transpose(hypthesis - y), (hypthesis - y))
    cost = (cost / 2 + Lambda*np.dot(np.transpose(theta),theta)/2)[0][0]
    # cost = (cost / 2)[0][0]
    MSE = math.sqrt(cost*2/X.shape[0])
    return MSE


def Drawlambda(X,X_train, y,exponent):

    # 迭代次数
    RangeLeft = -100
    RangeRight = 0
    ErmsStore = np.zeros(RangeRight-RangeLeft)
    LambdaStore = np.zeros(RangeRight-RangeLeft)
    for num in range(RangeLeft,RangeRight):

        LambdaStore[num-RangeLeft] = num/2
        Lambda = 10**(num/2)
        W = np.matmul(np.matmul(np.linalg.inv((np.matmul(X_train.T, X_train) + np.eye(exponent) * Lambda)), X_train.T),y)
        MSE = CalculateLambda(X_train,y,W,Lambda)

        # Cost = ComputeCost(X_train,y,theta,Lambda)

        ErmsStore[num-RangeLeft] = MSE
    title = 'exponent={} numbers={}'.format(exponent, X.shape[0])
    plt.title(title)
    plt.xlabel('$log_{10}\lambda$', fontsize=10)
    plt.ylabel('$MSE$', fontsize=10)
    plt.plot(LambdaStore, ErmsStore,'r.-')
    # print(LambdaStore)
    # print(LambdaStore.shape)
    # print(ErmsStore)
    # plt.legend()
    plt.savefig("./" + "picture/MSE/LS/" + title)
    plt.show()
