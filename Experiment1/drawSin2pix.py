import numpy as np
import matplotlib.pyplot as plt
def drawSin2pix(numbers):
    X = np.linspace(start=0, stop=1, num=numbers)
    y = np.sin(2 * np.pi * X)
    plt.plot(X,y,"m",label = "$y=sin(2*pi*x)$")