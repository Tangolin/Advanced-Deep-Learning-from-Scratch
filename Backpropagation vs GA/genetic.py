import scipy.io as sio
import numpy as np
import time

data= sio.loadmat('ex4data1.mat')
X= np.ones((data['X'].shape))
y= data['y']

for i in range(len(data['X'])):
    temp= data['X'][i].reshape((20,20)).T
    X[i]= temp.reshape(-1)

X= np.c_[np.ones(len(X),), data['X']]
m,n= X.shape
best = []

def relu(X):
    return X * (X > 0)

def softmax(X):
    exps = np.exp(X)
    return (exps.T / np.sum(exps, axis= 1)).T

def network(X, y, inputsize, layer1, layer2, output, theta):
    a = layer1 * (inputsize+1)
    b = a + layer2 * (layer1+1)
    c = b + output * (layer2+1)
    theta1 = theta[:a].reshape((layer1, inputsize+1))
    theta2 = theta[a:b].reshape((layer2, layer1+1))
    theta3 = theta[b:c].reshape((output, layer2+1))

    a_1 = np.dot(X, theta1.T)
    z_1 = np.c_[np.ones(m,),relu(a_1)]

    a_2 = np.dot(z_1, theta2.T)
    z_2 = np.c_[np.ones(m,),relu(a_2)]

    a_3 = np.dot(z_2, theta3.T)
    output = softmax(a_3)

    prediction = np.argmax(output, axis=1).reshape(m,1)
    accuracy = sum(prediction == y) / m

    return accuracy

def initialisation(epsilon, inputsize, layer1, layer2, output):
    total = layer1 * (inputsize+1) + layer2 * (layer1+1) + output * (layer2+1)
    theta = np.random.uniform(size = (population, total)) * 2 * epsilon - epsilon
    return theta

def fitness_f(X, y, inputsize, layer1, layer2, output, theta):
    fitness = np.ones(shape=(theta.shape[0],1))
    for i in range(len(theta)):
        fitness[i] = network(X, y, inputsize, layer1, layer2, output, theta[i])
    return fitness

def selection(population, fitness):
    population = np.c_[population, fitness]
    population = population[population[:,-1].argsort()[::-1]]
    best.append(population[0,-1])
    population = population[:50,:-1]
    return population

# def selection(population, fitness):
#     selected = np.zeros((200,population.shape[1]))
#     end = population.shape[0]
#     population = np.c_[population, fitness]
#     population = population[population[:,-1].argsort()[::-1]]
#     best.append(population[0,-1])
#     for i in range(200):
#         k = np.random.choice(np.arange(end), size=20, replace=False)
#         contestants = population[k,:]
#         contestants = contestants[contestants[:,-1].argsort()[::-1]]
#         winner = contestants[0]
#         selected[i] = winner[:-1]
#     return selected

def cross_over(selected):
    g_length = selected.shape[1]
    np.random.shuffle(selected)
    offspring = np.zeros((population, g_length))
    j=0
    for i in range(0,len(selected) -1,2):
        temp = np.sort(np.random.randint(2, size = (20,g_length)))
        pset = temp * selected[i] + 1*(temp==0) * selected[i+1]
        offspring[j:j+20,:] = pset
        j += 20
    return offspring

def mutate(m_const, offspring, epoch):
    substitute = np.random.uniform(-epsilon, epsilon, offspring.shape)
    a = np.random.choice([0,1], offspring.shape, p = [m_const+(epoch/10000), 1-(m_const+epoch/10000)])
    offspring = np.where(a==1,offspring, substitute)
    return offspring

input_size = 400
h1 = 26
h2 = 26
output_size = 10
epsilon = 0.12
m_int = 0.011
epoch = 500
population = 500

thetas = initialisation(epsilon, input_size, h1, h2, output_size)

for i in range(epoch):
    fitness_vector = fitness_f(X, y, input_size, h1, h2, output_size, thetas)
    chosen = selection(thetas, fitness_vector)
    thetas = cross_over(chosen)
    thetas = mutate(m_int, thetas, i+1)
    print(f'For generation {i+1}, the best performance is {float(max(best)):.2%}')