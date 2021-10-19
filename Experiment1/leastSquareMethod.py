import numpy as np
import matplotlib.pyplot as plt
from generateData import generatePlotdata
from drawSin2pix import drawSin2pix
from calculateLambda import *
from computeCost import *
def fittingNoRegular(X,X_train,y_noise,exponent):
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.T, X_train)), X_train.T), y_noise)

    # a = np.matmul(X_train.T, X_train)
    # ani = np.linalg.inv(a)
    # ji = np.matmul(a,ani)
    # print(a)

    # if np.linalg.cond(np.matmul(X_train.T, X_train)) > np.finfo(np.matmul(X_train.T, X_train).dtype).eps:
    #     print("病态矩阵")


    W = np.reshape(W, (exponent, 1))
    Y_predict = np.matmul(X_train, W)
    # plt.plot(X, Y_predict,format('b.-'))
    # 计算误差函数
    loss = ComputeCost(X_train,np.reshape(y_noise,(-1,1)),np.transpose(W),0)[0][0]
    print("loss = {}".format(loss))
    # print(W)
    # print(max(W))

    title = 'exponent={} numbers={}'.format(exponent,X.shape[0])
    plt.title(title)
    plt.xlabel('$X$', fontsize=10)
    plt.ylabel('$y$', fontsize=10)
    plt.scatter(X, y_noise, marker='o', label='Data Point')
    # numbers为绘图的点数
    X,X_train = generatePlotdata(numbers=1000,exponent=exponent)
    Y_predict = np.matmul(X_train, W)
    plt.plot(X,Y_predict,format('b'),label='Analytical Without Regularization')
    drawSin2pix(1000)
    plt.legend()
    # plt.savefig("./" + "picture/LS/NoRegular/" + title)
    plt.show()





def fittingRegular(X,X_train,y_noise,exponent,Lambda):
    W = np.matmul(np.matmul(np.linalg.inv((np.matmul(X_train.T, X_train) + np.eye(exponent) * Lambda)), X_train.T),y_noise)
    # W = np.reshape(W, (exponent, 1))
    Y_predict = np.matmul(X_train, W)
    # plt.plot(X, Y_predict)
    loss = ComputeCost(X_train, y_noise, np.transpose(W), 0)
    print("loss = {}".format(loss))
    title = 'exponent={} numbers={}'.format(exponent,X.shape[0])
    plt.title(title)
    plt.xlabel('$X$', fontsize=10)
    plt.ylabel('$y$', fontsize=10)
    plt.scatter(X, y_noise, marker='o', label='Data Point')
    # numbers为绘图的点数
    X, X_train = generatePlotdata(numbers=1000, exponent=exponent)
    Y_predict = np.matmul(X_train, W)
    plt.plot(X, Y_predict, format('b'), label='Analytical With Regularization')
    drawSin2pix(1000)
    plt.legend()
    # plt.savefig("./" + "picture/LS/Regular/" + title)
    plt.show()
