import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# open the .mat data file
data = scipy.io.loadmat("data/ex4data1.mat")
m,n = np.shape(data['X'])   # trainind data shape m=rows, n=features
X = data['X']
y = data['y'].reshape(-1)

num_labels = np.unique(y).size
y[np.where(y==10)] = 0
y = to_categorical(y)

forward_recurrence = 7
chromosomes = 50
num_neurons = 100
num_connections = 50
keep_best_n = 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



def ReLU(x):
    return x * (x>0)

def softmax(x):
    exps = np.exp(x)
    return (exps.T / np.sum(exps,axis=1)).T

def matrix_softmax(x):
    exps = np.exp(x)
    return (exps / np.sum(exps, axis=2)[:, :, np.newaxis])

def create_neuron_mask_as_tensor():
    """
    num_chromosomes, num_neurons, (n+num_neurons)
    """
    print("Creating new neuron mask tensor")
    epsilon_init = .12
    index_list = np.random.randint(0, (n+num_neurons), (chromosomes, num_neurons, num_connections))
    neuron_mask = np.zeros((chromosomes, num_neurons, n+num_neurons))
    for x in range(chromosomes):
        for y in range(num_neurons):
            if y < num_neurons-num_labels:
                neuron_mask[x][y][index_list[x][y]] = np.random.uniform(0,1,(num_connections))*2*epsilon_init - epsilon_init
            else:
                neuron_mask[x][y][-num_neurons:] = np.random.uniform(0,1,(num_neurons))*2*epsilon_init - epsilon_init
    return neuron_mask

def evaluate_neuron_vector(mask):
    neuron_structure = default_neuron_structure.copy()
    for _ in range(forward_recurrence):
        neuron_structure = np.c_[X.copy(), ReLU(np.dot(neuron_structure, mask.T))]
    return -np.sum(y*np.log(softmax(neuron_structure[:, -num_labels:])))

def evaluate_neuron_vector_accuracy(mask):
    neuron_structure = default_neuron_structure.copy()
    for _ in range(forward_recurrence):
        neuron_structure = np.c_[X.copy(), ReLU(np.dot(neuron_structure, mask.T))]
    return np.sum(np.argmax(softmax(neuron_structure[:, -num_labels:]),axis=1) == \
                  np.argmax(y, axis=1)\
                  ) * 100 / len(y)

def evaluate_neuron_tensor(generation):
    # shape: chromosomes, neurons, weight_mask
    neuron_structure_matrix = np.repeat(default_neuron_structure.copy()[np.newaxis, :, :], chromosomes, axis=0) # copy the 2d across 0-axis chromosomes times to create tensor
    transposed_generation = np.transpose(generation, (0,2,1))
    for _ in range(forward_recurrence):
        neuron_structure_matrix = np.c_[X_repeated_c.copy(), \
                                        ReLU(neuron_structure_matrix @ transposed_generation)]

    return (-np.sum(y*np.log(matrix_softmax(neuron_structure_matrix[:, :, -num_labels:])), axis=(1,2)))

def repopulation(fittest_matrix):
    mutation_range = .75
    length, num_neurons, weights = np.shape(fittest_matrix)

    p_index = np.random.randint(0,length, (2, chromosomes-length))
    rndm_mask = np.random.uniform(0,1, (chromosomes-length, num_neurons, weights)) >= .5
    mutation_mask = np.random.uniform(1-mutation_range, 1+mutation_range, (chromosomes-length, num_neurons, weights))

    new_generation = (fittest_matrix.copy()[p_index[0]] * rndm_mask + fittest_matrix.copy()[p_index[1]] * ~rndm_mask) * mutation_mask
    return np.concatenate((fittest_matrix, new_generation),axis=0)

def test_model(neuron_vector):
    loss = evaluate_neuron_vector(neuron_vector)
    accuracy = evaluate_neuron_vector_accuracy(neuron_vector)
    print(f"Loss:\t{loss:.2f}\tAccuracy:\t{accuracy}%")

def train():
    neuron_mask = create_neuron_mask_as_tensor()
    EPOCHS = 100
    keep_best_n = 10
    for epoch in range(EPOCHS):
        print(f"evaluating epoch: {epoch}")
        fitness_vector = evaluate_neuron_tensor(neuron_mask)

        n_th_val = np.sort(fitness_vector)[keep_best_n]
        best = np.argmin(fitness_vector)
        test_model(neuron_mask[best])
        neuron_mask = repopulation(fittest_matrix=neuron_mask[np.where(fitness_vector<=n_th_val)][:keep_best_n])


default_neuron_structure = np.zeros((m, num_neurons+n))
default_neuron_structure[:, :n] = X.copy()
X_repeated_c = np.repeat(X.copy()[np.newaxis, :, :], chromosomes, axis=0)
train()
