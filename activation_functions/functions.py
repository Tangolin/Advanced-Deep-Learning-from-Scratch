import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/ (1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def hard_sigmoid(x):
    return np.clip((x+1)/2, 0, 1)

def d_hard_sigmoid(x):
    x[np.where(np.logical_and(x>0, x<1))] = 1/2
    x[np.where(~np.logical_and(x>0, x<1))] = 0
    return x

def SiLU(x):
    # Sigmoid-Weighted Linear Units
    x *= sigmoid(x)
    return x

def d_SiLU(x):
    return sigmoid(x) + x * d_sigmoid(x)

def ReLU(x):
    return np.max(np.c_[np.zeros((np.shape(x)[0])), x], axis=1)

def d_ReLU(x):
    x[np.where(x<=0)] = 0
    x[np.where(x>0)] = 1
    return x

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def d_tanh(x):
    return 1 - tanh(x)**2

def hard_tanh(x):
    return np.clip(x, -1, 1)

def d_hard_tanh(x):
    x[np.where(np.logical_and(x>-1, x<1))] = 1
    x[np.where(np.logical_or(x<-1, x>1))] = 0
    return x

def softmax(x):
    exps = np.exp(x)
    return exps/np.sum(exps)

def d_softmax(x):
    return softmax(x) * (1-softmax(x))

def soft_sign(x):
    return x / (np.abs(x) + 1)

def d_soft_sign(x):
    """x[np.where(x<0)] = 1/((1-x[np.where(x<0)])**2)
    x[np.where(x==0)] = 0
    x[np.where(x>0)] = (1+ 2*x[np.where(x>0)]) / ((1+x)**2)"""

    x[np.where(x<0)] = (x[np.where(x<0)]-x[np.where(x<0)]**2) / ((1-x[np.where(x<0)])**2)
    x[np.where(x==0)] = 0
    x[np.where(x>0)] = (-x[np.where(x>0)]**2 - 1) / ((x[np.where(x>0)]+1)**2)
    return x

def plot(func, start=-5, end=5, step=0.05, hold_on=False):
    x = np.arange(start,end,0.05)
    plt.plot(x, func(x.copy()))
    if not hold_on:
        plt.show()



plot(soft_sign, hold_on=True)
plot(d_soft_sign)
