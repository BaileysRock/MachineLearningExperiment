import numpy as np
from math import log
from math import e

def computeCost(X, Y, theta, Lambda):
    cost = 0
    for i in range(0, X.shape[1]):
        cost = cost + (
        -Y[i] * np.dot(theta, np.reshape(X[i], (-1, 1))) + log(1 + np.exp(np.dot(theta, np.reshape(X[i], (-1, 1))))),
        e)[0]
    cost = cost + Lambda / 2 * np.dot(theta, np.reshape(theta, (-1, 1)))
    return cost
