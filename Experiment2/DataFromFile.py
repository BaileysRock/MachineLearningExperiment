import pandas as pd
import numpy as np
from GenerateData import addUnitColumn


def readFromFile(DocumentName):
    df = pd.read_csv(DocumentName)
    df.rename(columns={df.columns.array[df.columns.shape[0] - 1]: 'Predict'}, inplace=True)
    Y = df['Predict']
    X = df.drop('Predict', axis=1)
    X = np.array(X.values)
    X = np.reshape(X, (1, X.shape[0], X.shape[1]))
    X = addUnitColumn(X)
    Y = Y.values
    return X, Y
