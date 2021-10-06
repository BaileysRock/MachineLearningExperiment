from generateData import *
from gradientDescent import *
from conjugateGradientDescent import *
from leastSquareMethod import fittingNoRegular, fittingRegular

if __name__ == '__main__':
    # 训练样本个数
    numbers = 100
    # 噪声Sigma
    Sigma = 0.3
    # 多项式的阶数
    exponent = 51
    # 惩罚系数Lambda
    Lambda = 1e-7
    # 梯度下降学习率
    alpha = 0.01
    # 梯度下降迭代次数
    GDiterNum = 800000000
    # 共轭梯度下降迭代次数
    CGiterNum = 100
    # 迭代精度
    precision = 1e-8
    # 生成数据
    X, X_train, y, y_noise = addNoise(numbers, exponent, Sigma)

    fittingNoRegular(X, X_train, y_noise, exponent)
    fittingRegular(X, X_train, y_noise, exponent, Lambda)
    # Drawlambda(X, X_train, y_noise, exponent)
    ConjugateGradientDescent(X, X_train, y_noise, exponent, CGiterNum, Lambda, precision)
    GradientDescent(X, X_train, y_noise, exponent, alpha, GDiterNum, Lambda, precision)



# def model(numbers):
#     # numbers = 25
#     # # 多项式的阶数
#     exponent = 10
#     from sklearn import linear_model
#     model = linear_model.LinearRegression()
#     X, X_train, Y, Y_noise = addNoise(numbers, exponent)
#     X = np.reshape(X, (1, -1))
#     Y_noise = np.reshape(Y_noise, (1, -1))
#     model.fit(X, Y_noise)
#     Y_predict = model.predict(X)
#     X = np.reshape(X, (numbers, -1))
#     Y_predict = np.reshape(Y_predict, (numbers, -1))
#     plt.plot(X, Y_predict)
#     plt.show()
