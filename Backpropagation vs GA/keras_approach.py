import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt

import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential

# open the .mat data file
data = scipy.io.loadmat("data/ex4data1.mat")
m,n = np.shape(data['X'])   # trainind data shape m=rows, n=features
X = np.array(data['X'])#np.c_[np.ones(m), data['X']]
y = data['y'].reshape(-1)
y[np.where(y==10)] = 0
y = keras.utils.to_categorical(y)
#print(np.shape(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


print(np.shape(X_train))
print(np.shape(y_train))

"""handw = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = handw.load_data()
X_train /= 255.0
print(X_train)
X_test /= 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)"""

model = Sequential()
#model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile("SGD", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(X_train, y_train, epochs=25)
print(model.evaluate(X_test, y_test))
