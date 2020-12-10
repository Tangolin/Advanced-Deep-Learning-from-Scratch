import numpy as np
import matplotlib.pyplot as plt


def hard_sigmoid(x):
    #print(x)

    x = np.c_[np.ones(np.shape(x)[0]), x]
    print(np.min((x+1)/2, axis=1))
    return np.max( np.c_[np.zeros(np.shape(x)[0]), (np.min((x+1)/2, axis=1) ],axis=1)


def plot(func, start, end):
    x = np.arange(start, end, 0.01)
    plt.plot(x, func(x))
    plt.show()

plot(hard_sigmoid, -5, 5)
