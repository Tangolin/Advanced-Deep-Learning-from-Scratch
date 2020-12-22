import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt

from keras.utils import to_categorical

# open the .mat data file
data = scipy.io.loadmat("data/ex4data1.mat")
m,n = np.shape(data['X'])   # trainind data shape m=rows, n=features
X = np.c_[np.ones(m), data['X']] #np.array(data['X'])#np.c_[np.ones(m), data['X']]
y = data['y'].reshape(-1)
num_labels = np.unique(y).size
y[np.where(y==10)] = 0
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lambda_ = 1


def initialize_rndm_weights(l_in, l_out):
    """
    Input:
        l_in (integer): size of the input layer
        l_out (integer): size of the output layer
    Return:
        randomized wights with shape (l_out, l_in+1) #+1 to account for the bias unit
    """
    epsilon_init = .12
    return np.random.uniform(size=(l_out, 1+l_in)) * 2 * epsilon_init - epsilon_init

def softmax(x):
    #print(x)
    #i = input()
    exps = np.exp(x)
    return exps/np.sum(exps)

def d_softmax(x):
    return softmax(x) * (1-softmax(x))

def ReLU(x):
    return x * (x>0)

def d_ReLU(x):
    return (x>0)

def predict(X, theta_1, theta_2, theta_3):
    # Layer 1
    a_1 = np.dot(X, theta_1.T)
    z_1 = np.c_[np.ones(np.shape(a_1)[0],),ReLU(a_1) ]

    # Layer 2
    a_2 = np.dot(z_1, theta_2.T)
    z_2 = np.c_[np.ones(np.shape(a_2)[0],),ReLU(a_2) ]

    # Layer 3 - output
    a_3 = np.dot(z_2, theta_3.T)
    z_3 = softmax(a_3)
    return z_3


def compute_loss(X, y, theta_1, theta_2, theta_3):
    h = predict(X, theta_1, theta_2, theta_3)
    J = -np.sum(y*np.log(h), axis=1)
    return J

def compute_gradient(X, y, theta_1, theta_2, theta_3):
    # Layer 1
    a_1 = np.dot(X, theta_1.T)
    z_1 = np.c_[np.ones(np.shape(a_1)[0],),ReLU(a_1) ]

    # Layer 2
    a_2 = np.dot(z_1, theta_2.T)
    z_2 = np.c_[np.ones(np.shape(a_2)[0],),ReLU(a_2) ]

    # Layer 3 - output
    a_3 = np.dot(z_2, theta_3.T)
    z_3 = softmax(a_3)

    l_theta_1 = theta_1.copy()
    l_theta_1[:, 0] = 0
    l_theta_2 = theta_2.copy()
    l_theta_2[:, 0] = 0
    l_theta_3 = theta_3.copy()
    l_theta_3[:, 0] = 0


    delta_3 = z_3 - y
    #delta_2 = np.dot(delta_3, theta_3) * d_ReLU(np.c_[np.ones(np.shape(z_2)[0],) ,z_2])
    delta_2 = np.dot(delta_3, theta_3) * d_ReLU(z_2)
    delta_2 = delta_2[:, 1:]
    #delta_1 = np.dot(delta_2, theta_2) * d_ReLU(np.c_[np.ones(np.shape(z_1)[0],) ,z_1])
    delta_1 = np.dot(delta_2, theta_2) * d_ReLU(z_1)
    delta_1 = delta_1[:, 1:]

    Delta_1 = np.dot(delta_1.T, X)
    Delta_2 = np.dot(delta_2.T, np.c_[np.ones(np.shape(a_1)[0],), a_1])
    Delta_3 = np.dot(delta_3.T, np.c_[np.ones(np.shape(a_2)[0],), a_2])

    theta_1_grad = Delta_1/m + (lambda_/m) * l_theta_1
    theta_2_grad = Delta_2/m + (lambda_/m) * l_theta_2
    theta_3_grad = Delta_3/m + (lambda_/m) * l_theta_3

    return theta_1_grad, theta_2_grad, theta_3_grad


def train(X, y, theta_1, theta_2, theta_3):
    lr = 2e-3
    EPOCHS = 15
    iterations = 400
    for epoch in range(EPOCHS):
        for i in range(iterations):
            g_theta_1, g_theta_2, g_theta_3 = compute_gradient(X, y, theta_1, theta_2, theta_3)
            #print(g_theta_1)
            #print(g_theta_2)
            #print(g_theta_3)
            theta_1 -= lr*g_theta_1
            theta_2 -= lr*g_theta_2
            theta_3 -= lr*g_theta_3

        loss = compute_loss(X, y, theta_1, theta_2, theta_3)
        print(f"{epoch}:    Loss: {loss}")



def initialize_and_train(X, y):
    # learning variables
    input_layer_size = n
    hidden_layer_size_1 = 25
    hidden_layer_size_2 = 25

    # initialize weights
    theta_1 = initialize_rndm_weights(input_layer_size, hidden_layer_size_1)
    theta_2 = initialize_rndm_weights(hidden_layer_size_1, hidden_layer_size_2)
    theta_3 = initialize_rndm_weights(hidden_layer_size_2, num_labels)

    # train the Network
    theta_1, theta_2, theta_3 = train(X, y, theta_1, theta_2, theta_3)

    #return weights
    return theta_1, theta_2, theta_3


theta_1, theta_2, theta_3 = initialize_and_train(X=X_train, y=y_train)
compute_loss(X, y, theta_1, theta_2, theta_3)
#predict(X, theta_1, theta_2, theta_3)
