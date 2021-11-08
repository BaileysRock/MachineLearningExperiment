from PIL import Image
import numpy as np
import os


def readData(Path):
    """
    从文件中读取图像数据
    :return: 数据，数据的维度，数据的个数，文件目录
    """
    Data = []
    fileDirs = os.listdir(Path)
    for filename in fileDirs:
        image = Image.open(os.path.join(Path, filename)).convert("L")
        data = np.array(image, dtype=np.float32)
        Data.append(data)
    Data = np.array(Data)
    Dimension = Data.shape[2]
    Numbers = len(Data)
    return Data,Dimension,Numbers,fileDirs


def saveToFile(AfterProcessingData,FileDirs,DirPath):
    """
    保存到文件
    :param AfterProcessingData:处理后的数据
    :param PictureDimension: 图片的维度
    :param FileDirs: 文件的名字
    :return:
    """
    for i in range(AfterProcessingData.shape[0]):
        DataPiece = AfterProcessingData[i]
        im = Image.fromarray(DataPiece)
        im = im.convert('L')
        im.save(os.path.join(DirPath,"Processed_"+FileDirs[i]))



