from GenerateData import *
import matplotlib.pyplot as plt
from PCAModel import *

def PlotStraightLine(KEigenvalueVector):
    for i in range(0,1):
        X1 = [0]
        X2 = [0]
        if i ==0:
            for x in range(-8, 9):
                X1.append(x)
                X2.append(KEigenvalueVector[i][0] / KEigenvalueVector[i][1] * x)
        else:
            for x in range(-1, 2):
                X1.append(x)
                X2.append(KEigenvalueVector[i][0] / KEigenvalueVector[i][1] * x)
        plt.plot(X1,X2,label="the {} principal component".format(i))

def DealData():
    """
    使数据旋转
    :return:
    """
    Mu1 = 2
    Mu2 = 0
    cov11 = 10
    cov12 = 1
    cov21 = 1
    cov22 = 1
    X1,X2 = generate2DimensionalData(Mu1,Mu2,cov11=cov11,cov12=cov12,cov21=cov21,cov22=cov22,Num=100,noiseSigma1=0,noiseSigma2=0)

    plt.title("Mu1 = {},Mu2 ={},cov11 = {},cov12 = {},cov21 = {},cov22 = {}".format(Mu1,Mu2,cov11,cov12,cov21,cov22))
    Data = []
    Data.append(X1)
    Data.append(X2)
    Data = np.array(Data)
    FacePCAModel = PCAModel()
    KEigenvalueVector = np.transpose(FacePCAModel.fit(Data,2))
    PlotStraightLine(KEigenvalueVector)
    X = np.vstack((X1,X2))
    plt.scatter(X[0], X[1], marker='.', c='b', label="origin points")
    # plt.axis("scaled")
    plt.legend()
    plt.show()
    # 旋转
    KEigenvalueVector  = KEigenvalueVector.T
    temp = np.zeros(KEigenvalueVector[1].shape)
    temp = temp+KEigenvalueVector[0]
    KEigenvalueVector[0] = np.zeros(KEigenvalueVector[1].shape)
    KEigenvalueVector[0] = KEigenvalueVector[0] +KEigenvalueVector[1]
    KEigenvalueVector[1] = np.zeros(KEigenvalueVector[1].shape)
    KEigenvalueVector[1] = KEigenvalueVector[1] +temp
    KEigenvalueVector = KEigenvalueVector.T
    X = np.dot(KEigenvalueVector,X)
    plt.scatter(X[0],X[1],marker='.',c = 'b',label="rotation points")
    plt.axis("scaled")
    plt.legend()
    plt.show()


