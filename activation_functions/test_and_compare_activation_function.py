import numpy as np
import scipy.io
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt

from functions import *


def neural_network_mnist_classification(activation_func, d_activation_func):
    # open the .mat data file
    data = scipy.io.loadmat("data/ex4data1.mat")
    m,n = np.shape(data['X'])   # trainind data shape m=rows, n=features
    X = np.c_[np.ones(m), data['X']]
    y = data['y'].reshape(-1)


    # learning variables
    lambda_ = 1
    input_layer_size = n
    hidden_layer_size = 25
    num_labels = np.unique(y).size


    # define function & procedures
    def one_hot_encode_y(y):
        """ one-hot encode y e.g [1, 3, 5] = [[0,1,0...,0],[0,0,0,1,0,....0],...] """
        y = y-1
        one_hot_y = np.zeros((m,10))
        one_hot_y[np.arange(y.size), y] = 1
        return one_hot_y


    def decode_y(y):
        """ essentially the inverse of "one_hot_encode_y" """
        return np.argmax(y, axis=1)+1


    def plot_data(X,y,pred=[],amount=25, shuffle=True):
        if shuffle: # makes an initial call (before training) more diverse
            p = np.random.permutation(len(X))
            y, X = y[p], X[p]
        n_rows = int(np.floor(np.sqrt(amount)))
        n_cols = int(amount//n_rows)
        pred = [None for _ in range(len(y))] if len(pred)==0 else pred
        y, X, pred = y[:amount], X[:amount], pred[:amount]  # truncate if too long
        # plot images
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5*n_cols, 2*n_rows))
        for i in range(amount):
            ax = axes[i//n_cols, i%n_cols]
            ax.imshow(X[i].reshape(20,20).T, cmap='gray_r')
            ax.set_title('Label: {}\npred: {}'. format(y[i], pred[i]))
        plt.tight_layout()
        plt.show()


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


    def compute_cost_and_gradient(X, y, theta_1, theta_2, lambda_):
        """ This function should be used for the implementation of a training algorithm
        that doesn't use scipy. The loss return creates data for a useful plot """
        # Forward pass
        m = X.shape[0]
        z1 = np.dot(X, theta_1.T)
        a2 = np.c_[np.ones(np.shape(z1)[0]), activation_func(z1)]
        z2 = np.dot(a2, theta_2.T)
        h = activation_func(z2)

        l_theta_1 = theta_1.copy()
        l_theta_2 = theta_2.copy()
        l_theta_1[:, 0] = 0
        l_theta_2[:, 0] = 0
        #J = (1/m) * (np.sum(np.multiply(np.log(h), (-y))) - np.sum(np.multiply(np.log(1-h), (1-y))))
        #s = h-y
        J = (1/(2*m)) * np.sum((h-y)**2)
        J += lambda_/(2*m) * (np.sum(l_theta_1**2)+np.sum(l_theta_2**2)) # add the regularization penalty

        # compute the gradients for theta_1 and theta_2 via backpropagation
        delta_3 = h-y
        delta_2 = np.dot(delta_3, theta_2) * d_activation_func(np.c_[np.ones(np.shape(z1)[0]), z1])
        delta_2 = delta_2[:,1:]

        Delta_1 = np.dot(delta_2.T, X)
        Delta_2 = np.dot(delta_3.T, a2)

        theta_1_grad = Delta_1/m + (lambda_/m)*l_theta_1
        theta_2_grad = Delta_2/m + (lambda_/m)*l_theta_2
        return J, flatten_thetas(theta_1, theta_2)


    def flatten_thetas(theta_1, theta_2):
        """ joins and reduced dimension to 1 """
        return np.append(theta_1, theta_2).flatten()


    def un_flatten_theta(theta_list, input_layer_size, hidden_layer_size, num_labels):
        """ size specific inverse of "flatten_thetas" """
        theta_1 = theta_list[:(hidden_layer_size*(input_layer_size+1))]
        theta_2 = theta_list[(hidden_layer_size*(input_layer_size+1)):]
        return theta_1.reshape(hidden_layer_size, (input_layer_size+1)),\
              theta_2.reshape(num_labels, (hidden_layer_size+1))


    def compute_cost(theta_list, X, y, lambda_, input_layer_size, hidden_layer_size, num_labels):
        theta_1, theta_2 = un_flatten_theta(theta_list, input_layer_size, hidden_layer_size, num_labels)
        # Forward pass
        m = X.shape[0]
        z1 = np.dot(X, theta_1.T)
        a2 = np.c_[np.ones(np.shape(z1)[0]), activation_func(z1)]
        z2 = np.dot(a2, theta_2.T)
        h = activation_func(z2)

        l_theta_1 = theta_1.copy()
        l_theta_2 = theta_2.copy()
        l_theta_1[:, 0] = 0
        l_theta_2[:, 0] = 0
        #J = (1/m) * (np.sum(np.multiply(np.log(h), (-y))) - np.sum(np.multiply(np.log(1-h), (1-y))))
        J = (1/(2*m)) * np.sum((h-y)**2)
        J += lambda_/(2*m) * (np.sum(l_theta_1**2)+np.sum(l_theta_2**2))    # add the regularization penalty
        return J


    def compute_gradient(theta_list, X, y, lambda_, input_layer_size, hidden_layer_size, num_labels):
        theta_1, theta_2 = un_flatten_theta(theta_list, input_layer_size, hidden_layer_size, num_labels)
        # Forward pass
        m = X.shape[0]
        z1 = np.dot(X, theta_1.T)
        a2 = np.c_[np.ones(np.shape(z1)[0]), activation_func(z1)]
        z2 = np.dot(a2, theta_2.T)
        h = activation_func(z2)

        l_theta_1 = theta_1.copy()
        l_theta_2 = theta_2.copy()
        l_theta_1[:, 0] = 0
        l_theta_2[:, 0] = 0

        delta_3 = h-y
        delta_2 = np.dot(delta_3, theta_2) * d_activation_func(np.c_[np.ones(np.shape(z1)[0]), z1])
        delta_2 = delta_2[:,1:]

        Delta_1 = np.dot(delta_2.T, X)
        Delta_2 = np.dot(delta_3.T, a2)

        theta_1_grad = Delta_1/m + (lambda_/m)*l_theta_1
        theta_2_grad = Delta_2/m + (lambda_/m)*l_theta_2
        return flatten_thetas(theta_1_grad, theta_2_grad)


    def compute_numerical_gradient(theta_list, X, y, lambda_, input_layer_size, hidden_layer_size, num_labels):
        e = 1e-4
        numgrad = np.zeros(np.shape(theta_list))
        perturb = np.zeros(np.shape(theta_list))

        for p in range(0,np.size(theta_list)):
            perturb[p] = e
            loss_1 = compute_cost(theta_list-perturb, X, y, lambda_, input_layer_size, hidden_layer_size, num_labels)
            loss_2 = compute_cost(theta_list+perturb, X, y, lambda_, input_layer_size, hidden_layer_size, num_labels)
            numgrad[p] = (loss_2-loss_1) / (2*e)
            perturb[p] = 0
        return numgrad

    def check_neural_network_gradients(lambda_):
        """
        Checks via a reandomly generated training set whether the backpropagation
        algoithm is properly inmplemented.
        """
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 5
        theta_1 = initialize_rndm_weights(input_layer_size, hidden_layer_size)
        theta_2 = initialize_rndm_weights(hidden_layer_size, num_labels)
        # resuing initialize_rndm_weights to generate random X
        X = initialize_rndm_weights(input_layer_size-1, m)
        X = np.c_[np.ones(m), X]
        y = np.random.randint(0, 10, (m, num_labels))
        theta_list = flatten_thetas(theta_1, theta_2)

        cost = compute_cost(theta_list, X, y, lambda_, input_layer_size, hidden_layer_size, num_labels)
        grad = compute_gradient(theta_list, X, y, lambda_, input_layer_size, hidden_layer_size, num_labels)
        num_grad = compute_numerical_gradient(theta_list, X, y, lambda_, input_layer_size, hidden_layer_size, num_labels)
        #for i in range(np.size(grad)):
        #    print(f"grad: {grad[i]}\t\tnum_grad: {num_grad[i]}")

        difference = np.sum(np.absolute(grad-num_grad))
        #print(f"For this specific example, the distance should be smaller than 1e-9.")
        #print(f"Your distance is: {difference}")

    def train(theta_list, X, y, lambda_, input_layer_size, hidden_layer_size, num_labels):
        """ train the Neural Network via backpropagation and fmin_cg (scipy) """
        #print(f"Pre-training cost: {compute_cost(theta_list, X, y, lambda_, input_layer_size, hidden_layer_size, num_labels)}")
        result = scipy.optimize.fmin_cg(compute_cost, fprime=compute_gradient, x0=theta_list, \
                                        args=(X, y, lambda_, input_layer_size, hidden_layer_size, num_labels), \
                                        maxiter=50, disp=False, full_output=True )
        #print(f"Post-training cost: {result[1]}")
        return result[0]


    def predict(X, theta_1, theta_2):
        """ use the trained Neural Network to predict labels for mnist (output is one-hot) """
        z1 = np.dot(X, theta_1.T)
        a2 = np.c_[np.ones(np.shape(z1)[0]), activation_func(z1)]
        z2 = np.dot(a2, theta_2.T)
        a3 = activation_func(z2)
        return a3


    # initialize weights
    theta_1 = initialize_rndm_weights(input_layer_size, hidden_layer_size)
    theta_2 = initialize_rndm_weights(hidden_layer_size, num_labels)
    theta_list = flatten_thetas(theta_1, theta_2)
    y = one_hot_encode_y(y)
    check_neural_network_gradients(lambda_)
    theta_1,theta_2 = un_flatten_theta(train(theta_list, X, y, lambda_, input_layer_size, hidden_layer_size, num_labels), input_layer_size, hidden_layer_size, num_labels)

    # test accuracy
    test_amount = 2500
    p = np.random.permutation(len(X))[:test_amount]
    X_test, y_test = X[p], y[p]
    pred = predict(X_test, theta_1, theta_2)
    y_test, pred = decode_y(y_test), decode_y(pred)
    print(f"The accuracy on {test_amount} randomly chosen samples is \t\t{np.mean([pred==y_test])*100}%")
    #plot_data(X_test[:,1:],y_test,pred=pred,amount=np.min([test_amount, 49]), shuffle=False)




function_list = [[linear, d_linear, "Linear"],
                 [sigmoid, d_sigmoid, "Sigmoid"],
                 [hard_sigmoid, d_hard_sigmoid, "hard sigmoid"],
                 [SiLU, d_SiLU, "SiLU"],
                 [tanh, d_tanh, "tanh"],
                 [softmax, d_softmax, "softmax"],
                 [soft_sign, d_soft_sign, "soft_sign"],
                 [ReLU, d_ReLU, "ReLU"],
                 [LReLU, d_LReLU, "LReLU"],
                 [softplus, d_softplus, "softplus"],
                 [ELUs, d_ELUs, "ELUs"],
                 [swish, d_swish, "Swish"]]


for af, d_af, name in function_list:
    #i = input()
    print(name+" ", end=' ')
    neural_network_mnist_classification(activation_func=af,
                                        d_activation_func=d_af)
