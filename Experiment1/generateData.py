import numpy as np
def generateNoise(Sigma, Size):
    Noise = np.random.normal(0, scale=Sigma, size=Size)
    return Noise

def addNoise(numbers,exponent,Sigma):
    X = np.linspace(start=0, stop=1, num=numbers)
    GuassNoise = np.random.normal(0, scale=Sigma, size=X.shape)
    y = np.sin(2 * np.pi * X)
    y_noise = y + GuassNoise
    # plt.scatter(X, y_noise)
    # plt.show()
    row = np.ones(numbers, dtype=np.float64) * X
    X_train = row ** 0
    # 计算不同阶对应X值
    for i in range(1, exponent):
        row = np.ones(numbers, dtype=np.float64) * X
        row = row ** i
        X_train = np.dstack((X_train, row))
    X_train = np.reshape(X_train, (numbers, exponent))
    return X,X_train,y, y_noise


def generatePlotdata(numbers,exponent):
    X = np.linspace(start=0, stop=1, num=numbers)
    row = np.ones(numbers, dtype=np.float64) * X
    X_train = row ** 0
    # 计算不同阶对应X值
    for i in range(1, exponent):
        row = np.ones(numbers, dtype=np.float64) * X
        row = row ** i
        X_train = np.dstack((X_train, row))
    X_train = np.reshape(X_train, (numbers, exponent))
    return X,X_train