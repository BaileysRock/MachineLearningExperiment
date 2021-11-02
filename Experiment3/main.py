from KMeans import *
from generateData import *
from GMM import *
from readFromFile import *

if __name__ == '__main__':
    # # 类别个数
    # K = 3
    # # 迭代结束次数
    # iter = 200
    # # 生成数据
    # X1, X2 = generateData(K)
    # KMeansClusterData = makeDataForKmeans(X1, X2)
    # GMMClusterData = makeDataForGMM(X1, X2)
    # # KMeans算法
    # DataCenter, centers = kMeans(K, iter, KMeansClusterData)
    # drawKmeansClassify(KMeansClusterData, DataCenter, centers, K)
    # # GMM EM算法
    # Gamma, mus,sigmas,alpha = GMMem(K, iter, GMMClusterData)
    # drawGMM(GMMClusterData, Gamma, mus)



    iterFile = 200
    # 从文件中读取数据
    X, Y, K = readFromFile("./DataSet/iris.csv")
    KMeansTrain, KMeansTest, KMeansYTrain, KMeansYtest = makeFileDataForKmeans(X, Y)
    GMMTrain, GMMTest, GMMYTrain, GMMYTest = makeFileDataForGMM(X, Y)

    # KMeans算法
    DataCenterFile, centersFile = kMeans(K, iterFile, KMeansTrain)
    TestKMeansAccuracy(DataCenterFile,centersFile,KMeansYTrain,KMeansYtest,Y,KMeansTest,K)


    # GMM EM算法
    GammaFile, mus,sigmas,alpha = GMMem(K, iterFile, GMMTrain)
    centersTrain = np.zeros(GammaFile.shape[0])
    for i in range(GammaFile.shape[0]):
        centersTrain[i] = np.array(GammaFile[i]).argmax()
    TestGMMAccuracy(centersTrain,mus,sigmas,alpha,GMMYTrain,GMMYTest,Y,GMMTest,K)
