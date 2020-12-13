import numpy as np
import matplotlib.pyplot as plt

"""
Allows for scalar and vector inputs (some also for matrix)
"""

def linear(x):
    return x

def d_linear(x):
    return 1

def sigmoid(x):
    return 1/ (1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def hard_sigmoid(x):
    #return np.clip((x+1)/2, 0, 1)
    return np.clip((x+1)/2, 0.001, 0.999)

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
    return x / (np.abs(x)+1)**2

def ReLU(x):
    return x * (x>0)

def d_ReLU(x):
    x = x.copy()
    x[np.where(x<=0)] = 0
    x[np.where(x>0)] = 1
    return x

def LReLU(x, a=0.01):
    # Leaky ReLU
    x[np.where(x<=0)] = a*x[np.where(x<=0)]
    return x

def d_LReLU(x, a=0.01):
    x[np.where(x>0)] = 1
    x[np.where(x<=0)] = a
    return x
def PReLU(x):
    # Parametric Rectified Linear Units
    #### learnable parameter is slightly harder to implement, eh
    return x

def d_PReLU(x):
    # same walao
    return x

def RPReLU(x):
    # Randomized Leaky Recitified Linear Units
    ### ayio, learning parameter
    return x

def d_RPReLU(x):
    # same eh
    return x

def SReLU(x):
    # S-shaped Rectified Linear Units
    #same learning
    return x

def d_SReLU(x):
    # learned parameters
    return x

def softplus(x):
    return np.log(1+np.exp(x))

def d_softplus(x):
    return np.exp(x) / (1+np.exp(x))

def ELUs(x, a=1):
    # Exponential Linear Units
    x[np.where(x<=0)] = a*np.exp(x[np.where(x<=0)])-1
    return x

def d_ELUs(x, a=1):
    x[np.where(x>0)] = 1
    x[np.where(x<=0)] = ELUs(x[np.where(x<=0)].copy()) + a
    return x

def PELU(x, a=1, b=1, c=1):
    # Parametric Exponential Linear Unit                        ##### not working (1,1,1) should return ELU
    x[np.where(x>0)] = c*x[np.where(x>0)]
    x[np.where(x<=0)] = a*np.exp(x[np.where(x<=0)]/b) #updated
    return x

def d_PELU(x, a, b, c):
    x[np.where(x>0)] = c
    x[np.where(x<=0)] = (a/b)*np.exp(x[np.where(x<=0)]/b)
    return x

def SELU(x, a=1.6733, r=1.0507):
    # Scaled Exponential Linear Units
    x[np.where(x<=0)] = a*np.exp(x[np.where(x<=0)])-a
    return r*x

def d_SELU(x, a=1.6733, r=1.0507):
    x[np.where(x>0)] = 1
    x[np.where(x<=0)] = a*r*np.exp(x[np.where(x<=0)])
    return x

def maxount(x):
    # requires a network for implementation
    return x

def d_maxount(x):
    return x

def swish(x):
    return x * sigmoid(x)

def d_swish(x):
    return sigmoid(x) * (x * d_sigmoid(x)) #updated

def ELiSH(x):
    # Exponential linear Squashing
    x[np.where(x>=0)] = swish(x[np.where(x>=0)])
    x[np.where(x<0)] = (np.exp(x[np.where(x<0)])-1) / (1+np.exp(-x[np.where(x<0)]))
    return x

def d_ELiSH(x):
    x[np.where(x>=0)] = d_swish(x[np.where(x>=0)])
    x[np.where(x<0)] = -2 * d_sigmoid(x)  #updated
    return x

def Hard_ELiSH(x):
    # Hard Exponential linear Squashing
    x[np.where(x>=0)] *= hard_sigmoid(x[np.where(x>=0)])
    x[np.where(x<0)] = (np.exp(x[np.where(x<0)])-1) * hard_sigmoid(x[np.where(x<0)])
    return x

def d_Hard_ELiSH(x):
    x[np.where(x>=0)] = hard_sigmoid(x[np.where(x>=0)]) + x[np.where(x>=0)]*d_hard_sigmoid(x[np.where(x>=0)])
    x[np.where(x<0)] = np.exp(x[np.where(x<0)]) * hard_sigmoid(x[np.where(x<0)]) + (np.exp(x[np.where(x<0)])-1) * d_hard_sigmoid(x[np.where(x<0)])
    # again, let's check my math on that
    return x

def plot(func, start=-5, end=5, step=0.05, hold_on=False, shape=False):
    x = np.arange(start,end,0.05)
    if shape:
        print(x.shape)
    plt.plot(x, func(x.copy()))
    if not hold_on:
        plt.show()

    if shape:
        print(func(x.copy()).shape)



#plot(ReLU, hold_on=False, shape=True)
