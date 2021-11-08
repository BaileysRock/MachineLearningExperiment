from PCAModel import *
from DealData import *
import matplotlib.pyplot as plt

def cal_psnr(im1, im2):
    diff = im1 - im2
    mse = np.mean(np.square(diff))
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr

def Face(K):
    """
    处理Face文件夹
    :param K:PCA降维后的维度
    :return:
    """
    # 读取人脸图片
    path = "./Face"
    Data,Dimension,Numbers,FileDirs = readData(path)
    ReconsitutionData = []
    # 训练PCA模型
    for i in range(Data.shape[0]):
        # 初始化PCA模型
        FacePCAModel = PCAModel()
        FacePCAModel.fit(Data[i],K)
        # 得到PCA处理后重构的数据
        ReconsitutionData.append(FacePCAModel.Reconsitution())
    ReconsitutionData = np.array(ReconsitutionData)
    # 保存到文件
    Dirpath = "./ProcessedFace"
    saveToFile(ReconsitutionData,FileDirs,Dirpath)
    for len in range(Data.shape[0]):
        path = "./Face/2.jpg"
        image = Image.open(path).convert("L")
        image1 = np.array(Data[len],dtype=np.uint8)
        image2 = np.array(ReconsitutionData[len],dtype=np.uint8)
        psnr = cal_psnr(image1,image2)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(ReconsitutionData[len])
        plt.title("dim = " + str(K) + "\nPSNR = " + str(psnr))
        plt.show()




def FaceData(K):
    """
    处理FaceData文件夹
    :param K:PCA降维后的维度
    :return:
    """
    # 读取人脸图片
    path = "./FaceData"
    Data, Dimension, Numbers, FileDirs = readData(path)
    ReconsitutionData = []
    # 训练PCA模型
    for i in range(Data.shape[0]):
        # 初始化PCA模型
        FacePCAModel = PCAModel()
        FacePCAModel.fit(Data[i], K)
        # 得到PCA处理后重构的数据
        ReconsitutionData.append(FacePCAModel.Reconsitution())
    ReconsitutionData = np.array(ReconsitutionData)
    # 保存到文件
    Dirpath = "./ProcessedFaceData"
    saveToFile(ReconsitutionData, FileDirs, Dirpath)
    for i in range(10):
        plt.figure()
        psnr = cal_psnr(Data[i],ReconsitutionData[i])
        plt.subplot(1,2,1)
        plt.imshow(Data[i])
        plt.subplot(1,2,2)
        plt.imshow(ReconsitutionData[i])
        plt.title("dim = " + str(K) + "\nPSNR = " + str(psnr))
        plt.show()

def FaceDataImprove(K):
    """
    处理FaceData文件夹
    :param K:PCA降维后的维度
    :return:
    """
    # 读取人脸图片
    path = "./FaceData"
    Data, Dimension, Numbers, FileDirs = readData(path)
    DataList = []
    for i in range(Data.shape[0]):
        DataVector = np.empty(0)
        # 转为行向量
        DataTranspose = np.transpose(Data[i])
        for j in range(Data[i].shape[1]):
            DataVector= np.hstack((DataVector,DataTranspose[j]))
        DataList.append(DataVector)
    DataArray = np.array(DataList)
    DataArray = np.transpose(DataArray)
    FacePCAModel = PCAModel()
    FacePCAModel.fit(DataArray, K)
    ReconsitutionData = FacePCAModel.Reconsitution()
    # 将每一列改为行
    ReconsitutionData = np.transpose(ReconsitutionData)
    ImagesList = []
    for i in range(ReconsitutionData.shape[0]):
        ImagesList.append(np.transpose(np.reshape(ReconsitutionData[i],(Dimension,-1))))
    ImagesArray = np.array(ImagesList,dtype=np.uint8)
    # ImagesArray = np.array(ImagesList)
    # 保存到文件
    Dirpath = "./ProcessedFaceDataImprove"
    # saveToFile(ImagesArray , FileDirs, Dirpath)
    for i in range(4):
        plt.figure()
        psnr = cal_psnr(Data[i],ImagesArray[i])
        plt.subplot(1,2,1)
        plt.imshow(Data[i])
        plt.subplot(1,2,2)
        plt.imshow(ImagesArray[i])
        plt.title("dim = " + str(K) + "\nPSNR = " + str(psnr))
        plt.show()