import numpy as np
from computeCost import ComputeCost
import matplotlib.pyplot as plt
from generateData import generatePlotdata
from drawSin2pix import drawSin2pix
def GradientDescent(X,X_train,y,exponent,alpha,iterNum,Lambda,precision):
    # 用于存储代价值
    costStore = np.zeros(iterNum)
    # 定义X数据的大小
    Xsize = X_train.shape[0]
    # 设置theta初始值
    theta = np.zeros(exponent)
    costStore[0] = ComputeCost(X_train, y, theta, Lambda)
    predict = np.matmul(X_train, theta.T)
    theta = theta - alpha * np.dot(X_train.T, (predict - y).T) / Xsize
    print("当前迭代的alpha={}".format(alpha))
    for num in range(1,iterNum):
        # 计算当前代价值并保存
        costStore[num] = ComputeCost(X_train,y,theta,Lambda)
        predict = np.matmul(X_train,theta.T)
        theta = theta - alpha *np.dot(X_train.T,(predict-y).T)/Xsize
        if abs(costStore[num]-costStore[num-1]) <= precision :
            print("迭代次数：{}".format(num))
            print("当前迭代的alpha={}".format(alpha))
            break
        if costStore[num]-costStore[num-1] >0:
            alpha = alpha/2
            print("当前迭代的alpha={}".format(alpha))
    Y_predict = np.matmul(X_train, theta)
    # plt.plot(X, Y_predict,format('y'))
    loss = ComputeCost(X_train, np.reshape(y,(-1,1)), np.transpose(theta), 0)[0][0]
    print("loss = {}".format(loss))

    # # numbers为绘图的点数
    # X,X_train = generatePlotdata(numbers=100,exponent=exponent)
    # Y_predict = np.matmul(X_train, theta)
    # plt.plot(X, Y_predict, format('y'),label = 'Gradient Descent')

    title = 'exponent={} numbers={} alpha={} lambda={}'.format(exponent,X.shape[0],alpha,Lambda)
    plt.title(title)
    plt.xlabel('$X$', fontsize=10)
    plt.ylabel('$y$', fontsize=10)
    plt.scatter(X, y, marker='o', label='Data Point')
    # numbers为绘图的点数
    X, X_train = generatePlotdata(numbers=1000, exponent=exponent)
    Y_predict = np.matmul(X_train, theta)
    plt.plot(X, Y_predict,format('c'),label='Gradient Descent')
    drawSin2pix(1000)
    plt.legend()
    # plt.savefig("./" + "picture/GradientDescent/" + title+".png")
    plt.show()
