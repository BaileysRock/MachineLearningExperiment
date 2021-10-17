from PredictResult import predictResult

if __name__ == '__main__':
    # 使用生成的数据点求解
    # 符合朴素贝叶斯分布的点求解
    # predictResult(0)
    # 不符合朴素贝叶斯分布的点求解
    # predictResult(1)
    # 使用梯度下降法对保存的文件 求解
    predictResult(2,"./DataSet/data_banknote_authentication.csv")
    # predictResult(2,"./DataSet/sonar.csv")
    # predictResult(2, "./DataSet/heart.csv")
    # 使用牛顿迭代法对保存的文件 求解
    predictResult(3, "./DataSet/data_banknote_authentication.csv")
    # predictResult(3, "./DataSet/sonar.csv")
    # predictResult(3, "./DataSet/heart.csv")
