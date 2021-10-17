import numpy as np


def predictPrecision(theta, X_test, Y_test):
    """
    计算预测的准确率
    :param theta:迭代出来的系数
    :param X_test: 待测试数据
    :param Y_test: 数据集所属真实分类
    """

    predict = np.dot(X_test, theta)
    accuracy = 0
    for i in range(0, predict.shape[0]):
        truth = Y_test[i]
        if (predict[i] >= 0 and truth == 1) or (predict[i] <= 0 and truth == 0):
            accuracy += 1
    print("测试集总数{}，预测正确个数{}".format(predict.shape[0], accuracy))
    print("正确率{}".format(accuracy / predict.shape[0]))
