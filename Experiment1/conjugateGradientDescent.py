import numpy as np
from generateData import generatePlotdata
import matplotlib.pyplot as plt
from drawSin2pix import drawSin2pix
def ConjugateGradientDescent(X, X_train, y, exponent, CGiterNum, Lambda, precision):
    A = np.dot(np.transpose(X_train), X_train) + np.eye(exponent) * Lambda
    b = np.dot(np.transpose(X_train), y)
    b = np.reshape(b,(exponent,-1))
    Wk = np.ones(exponent)*0
    Wk = np.reshape(Wk,(exponent,-1))

    test = np.dot(A, Wk)

    rk = b - np.dot(A, Wk)
    rk = np.reshape(rk,(exponent,-1))
    Pk = rk
    precisionMatrix = np.ones(exponent)*precision
    for num in range(0, CGiterNum):

        alpha = ((np.dot(np.transpose(rk),rk))/(np.dot((np.dot(np.transpose(Pk),A)),Pk)))[0][0]
        Wk = Wk + alpha*Pk
        rkadd1 = rk - alpha*np.dot(A,Pk)
        if np.all(rkadd1 <= precisionMatrix):
            print("第{}次迭代收敛".format(num))
            break
        beta = (np.dot(np.transpose(rkadd1),rkadd1)/np.dot(np.transpose(rk),rk))[0][0]
        Pk = rkadd1+beta*Pk
        rk = rkadd1

    title = 'exponent={} numbers={} lambda={}'.format(exponent,X.shape[0],Lambda)
    plt.title(title)
    plt.xlabel('$X$', fontsize=10)
    plt.ylabel('$y$', fontsize=10)
    plt.scatter(X, y, marker='o', label='Data Point')
    # numbers为绘图的点数
    X, X_train = generatePlotdata(numbers=1000, exponent=exponent)
    Y_predict = np.matmul(X_train, Wk)
    plt.plot(X, Y_predict, format('c'), label='Conjugate Gradient Descent')
    drawSin2pix(1000)
    plt.legend()
    # plt.savefig("./" + "picture/ConjugateGradientDescent/" + title)
    plt.show()