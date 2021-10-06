import numpy as np
# 计算代价函数
def ComputeCost(X, y, theta,Lambda):
    hypthesis = np.dot(X, np.transpose(theta))
    # 先转置再做矩阵乘法
    cost = np.dot(np.transpose(hypthesis - y), (hypthesis - y))
    cost = cost / 2 + Lambda*np.dot(theta,np.transpose(theta))/2
    return cost

