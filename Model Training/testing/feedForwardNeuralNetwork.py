import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix

data = pd.read_csv('final.csv')

x = data.drop('Class', axis = 1)
y = data['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

model = Sequential()
for i in range (100):
	model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation=tf.keras.activations.sigmoid))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=1000)

print("evaluate: ", model.evaluate(x_test, y_test, verbose=0))

pred = [0 if x <= 0.6 else 1 for x in model.predict(x_test)]

print(pred)
print(y_test)

print(confusion_matrix(y_test, pred))


model.save("feedForwardNeuralNetwork.h5")

