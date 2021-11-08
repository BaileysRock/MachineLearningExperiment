import numpy as np

class PCAModel:
    def __init__(self) -> None:
        """
        对其初始化
        """
        super().__init__()
        self.Data = np.empty(0)
        self.K = 0
        self.CovarianceMatrix = np.empty(0)
        self.KEigenvalueVector = np.empty(0)
        self.Average = np.empty(0)
        self.AfterProcessingData = np.empty(0)

    def zeroValue(self):
        """
        将给定的数据进行零值化
        :return:
        """
        averageList = []
        Data = self.Data
        for i in range(Data.shape[0]):
            sum = np.sum(Data[i])
            average = sum/Data.shape[1]
            averageList.append(average)
            for j in range(Data.shape[1]):
                Data[i][j] = Data[i][j]-average
        self.Data = Data
        self.Average = np.array(averageList)

    def generateCovarianceMatrix(self):
        """
        生成协方差矩阵
        :return:
        """
        N = self.Data.shape[1]
        CovarianceMatrix = np.zeros((self.Data.shape[0],self.Data.shape[0]))
        Data = np.transpose(self.Data)
        for i in range(self.Data.shape[1]):
            CovarianceMatrix = CovarianceMatrix+np.dot(np.reshape(Data[i],(-1,1)),np.reshape(Data[i],(1,-1)))
        CovarianceMatrix = CovarianceMatrix/(N-1)
        self.CovarianceMatrix = CovarianceMatrix


    def takeKArray(self):
        """
        取前K行作为矩阵
        :return:
        """
        Eigenvalue,EigenvalueVector = np.linalg.eig(self.CovarianceMatrix)
        ArgSort = np.argsort(Eigenvalue)
        VectorList = []
        for i in range(self.K):
            for j in range(Eigenvalue.shape[0]):
                if ArgSort[j] == i:
                    VectorList.append(EigenvalueVector[j])
        self.KEigenvalueVector = np.transpose(np.array(VectorList))


    def AfterProcessing(self):
        """
        降维后的数据
        :return:返回降维后的数据
        """
        AfterProcessingData = np.dot(self.KEigenvalueVector.T,self.Data)
        # for i in range(returnData.shape[0]):
        #     for j in range(returnData.shape[1]):
        #         returnData[i][j] = returnData[i][j]+self.Average[i]
        self.AfterProcessingData = AfterProcessingData

    def fit(self,Data,K):
        """
        对给定的数值适配模型
        :param Data: 给定的数据
        :param K: 降维后的数据的维度
        :return:
        """
        self.Data = Data
        self.K = K
        assert self.Data.shape[0] >= self.K
        self.zeroValue()
        self.generateCovarianceMatrix()
        self.takeKArray()
        return self.KEigenvalueVector

    def Reconsitution(self):
        self.AfterProcessing()
        ReconsitutionData = np.dot(self.KEigenvalueVector,self.AfterProcessingData)
        # ReconsitutionData = ReconsitutionData +self.Average
        for i in range(ReconsitutionData.shape[0]):
            for j in range(ReconsitutionData.shape[1]):
                ReconsitutionData[i][j] = ReconsitutionData[i][j]+self.Average[i]
        return ReconsitutionData












