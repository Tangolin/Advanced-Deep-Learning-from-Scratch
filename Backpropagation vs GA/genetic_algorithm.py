import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# open the .mat data file
data = scipy.io.loadmat("data/ex4data1.mat")
m,n = np.shape(data['X'])   # trainind data shape m=rows, n=features
X = np.c_[np.ones(m), data['X']]
y = data['y'].reshape(-1)

num_labels = np.unique(y).size
lambda_ = 3

y[np.where(y==10)] = 0
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def initialize_rndm_weights_struct(l_in, l_out):
    epsilon_init = .12
    return np.random.uniform(size=(l_out, 1+l_in)) * 2 * epsilon_init - epsilon_init

def initialize_rndom_weights_flat(weight_size):
    epsilon_init = .12
    return np.random.uniform(size=(weight_size,)) * 2 * epsilon_init - epsilon_init

def initialize_rndom_weights_matrix(weight_size, chromosomes):
    epsilon_init = .12
    return np.random.uniform(size=(chromosomes, weight_size)) * 2 * epsilon_init - epsilon_init

def softmax(x):
    exps = np.exp(x)
    return (exps.T / np.sum(exps,axis=1)).T

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
    return -np.sum(y*np.log(h))

def compute_loss_list(X, y, theta_list):
    theta_1, theta_2, theta_3 = unflatten_thetas(theta_list)
    return compute_loss(X, y, theta_1, theta_2, theta_3)

def flatten_thetas(theta_1, theta_2, theta_3):
    return np.concat(np.concat(theta_1.reshape(-1), theta_2.reshape(-1)), theta_3.reshape(-1))

def unflatten_thetas(theta_flat):
    theta_1_range = 25 * 401
    theta_2_range = 25 * 26
    theta_3_range = 10 * 26

    theta_1 = theta_flat[0:theta_1_range].reshape(25,401)
    theta_2 = theta_flat[theta_1_range:theta_1_range+theta_2_range].reshape(25,26)
    theta_3 = theta_flat[theta_1_range+theta_2_range:theta_1_range+theta_2_range+theta_3_range].reshape(10,26)
    return theta_1, theta_2, theta_3

def evaluate_flat(theta_list):
    theta_1, theta_2, theta_3 = unflatten_thetas(theta_list)
    pred = np.argmax(predict(X_test, theta_1, theta_2, theta_3),axis=1)
    return  np.sum(pred==np.argmax(y_test,axis=1))/len(y_test)

def repopulation(fittest_matrix,chromosomes,best_n):
    mutation_range = 0.75
    length, elements = np.shape(fittest_matrix)
    r_1_start, r_1_end = chromosomes-best_n*2+1, chromosomes-best_n+1
    r_2_start, r_2_end = chromosomes-best_n, chromosomes+1
    new_generation = np.zeros((chromosomes, 10_935))

    new_generation[:best_n] = fittest_matrix
    new_generation[r_1_start:r_1_end] = fittest_matrix
    new_generation[r_2_start:r_2_end] = fittest_matrix
    for x in range(best_n, r_1_start):
        # mutation
        mutation_mask = np.random.uniform(1-mutation_range,1+mutation_range,elements)
        p_index = np.random.randint(0,length, (2,))
        rndm_mask = np.random.uniform(0,1, elements) >= .5
        new_generation[x] = (fittest_matrix[p_index[0]] * rndm_mask + fittest_matrix[p_index[1]] * ~rndm_mask)*mutation_mask
    for x in range(r_1_start, chromosomes):
        # mutation
        mutation_mask = np.random.uniform(1-mutation_range,1+mutation_range,elements)
        new_generation[x] *= mutation_mask

    return new_generation

def train(X, y, theta_matrix, chromosomes):
    EPOCHS = 150
    keep_best_n = 15
    for epoch in range(EPOCHS):
        fitness_vector = np.zeros((np.shape(theta_matrix)[0],))
        for x in range(np.shape(theta_matrix)[0]):
            # assess loss
            fitness_vector[x] = compute_loss_list(X, y, theta_matrix[x])

        # pass the best x thetas into the repopulation function
        # find the n_th value and take it as cut-off point
        n_th_val = np.sort(fitness_vector)[keep_best_n]
        best = np.argmin(fitness_vector)
        print(f"{epoch}:\tloss:{compute_loss_list(X, y, theta_matrix[best])}\taccuracy:{evaluate_flat(theta_matrix[best])}")
        theta_matrix = repopulation(fittest_matrix=theta_matrix[np.where(fitness_vector<=n_th_val)][:keep_best_n],
                                    chromosomes=chromosomes,
                                    best_n=keep_best_n)


def initialize_and_train(X, y):
    # learning variables
    input_layer_size = n
    hidden_layer_size_1 = 25
    hidden_layer_size_2 = 25
    chromosomes=200

    # initialize weights
    theta_matrix = initialize_rndom_weights_matrix(weight_size=401*25+25*26+10*26,
                                               chromosomes=chromosomes)

    # train the Network
    theta_1, theta_2, theta_3 = train(X=X,
                                      y=y,
                                      theta_matrix=theta_matrix,
                                      chromosomes=chromosomes)

    #return weights
    return theta_1, theta_2, theta_3


theta_1, theta_2, theta_3 = initialize_and_train(X=X_train, y=y_train)
# gets to about
# Accuracy  0.353
# Loss      7834.092141686689
# After 50 epochs
