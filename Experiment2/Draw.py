import matplotlib.pyplot as plt
import numpy as np


def drawTheta(Start, End, theta, method):
    # number
    number = 1000
    X = np.linspace(start=Start, stop=End, num=number)
    X = np.reshape(X, (-1, 1))
    UnitMatrix = np.reshape(np.ones(number), (-1, 1))
    Y = (theta[0] * UnitMatrix + theta[1] * X) * (-1) * (1 / theta[2])
    if method == 0:
        plt.plot(X, Y, 'g', label="Gradient Descent")
    elif method == 1:
        plt.plot(X, Y, 'g', label="Newton Method")
    plt.xlabel('$X_{1}$', fontsize=10)
    plt.ylabel('$x_{2}$', fontsize=10)
    plt.legend()
    plt.show()


def plotScatter(X1Y0, X2Y0, X1Y1, X2Y1):
    plt.scatter(X1Y0, X2Y0, label='Y{}'.format(0))
    plt.scatter(X1Y1, X2Y1, label='Y{}'.format(1))
