'''
References:
## https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
## https://www.youtube.com/watch?v=ftMq5ps503w
'''

from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


### read data
filename = 'sp500.csv'
data = open(filename, 'r').read().split()
result = []


### normalise dataset
data = np.array(data, dtype='float64')
data = np.reshape(data, (-1,1))
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
data = np.reshape(data, (-1,))

sequence_length = 50 + 1


### format data
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])

arr = np.asarray(result, dtype='float64')
size = arr.shape[0]

arr = np.expand_dims(arr, axis=1)


### train-test split
train_size = int(size*0.8)
train, test = arr[0:train_size], arr[train_size:]

xtrain = train[:,:,0:-1]
ytrain = train[:,:,-1:]
ytrain = ytrain[:,-1]

xtest = test[:,:,0:-1]
ytest = test[:,:,-1:]
ytest = ytest[:,-1]


### model
model = Sequential()

model.add(LSTM(50, input_shape=(1,sequence_length-1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')
model.summary()
model.fit(xtrain, ytrain, batch_size=64, epochs=5,
          validation_split=0.05, verbose=2)


### make prediction
tpred = model.predict(xtest)


### plot data
tpred = scaler.inverse_transform(tpred)
tpred = np.reshape(tpred, (len(tpred),))

ttrue = scaler.inverse_transform(ytest)
ttrue = np.reshape(ttrue, (len(tpred),))

plt.plot(tpred)
plt.plot(ttrue)
plt.show()
