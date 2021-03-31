import dataProcessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM 
from tensorflow.keras.layers import Dropout

def main():
	test = dataProcessing.readData()
	test = dataProcessing.cleanData(test)
	
	print(test[0])
	
	sc = MinMaxScaler(feature_range = (0, 1))
	trainingSetScaled = sc.fit_transform(test)

	x_train = []
	y_train = []
	for i in range(50, len(test)):
		x_train.append(trainingSetScaled[i - 50:i, 0])
		y_train.append(trainingSetScaled[i, 0])
	x_train, y_train = np.array(x_train), np.array(y_train)

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

	#Create model for learning
	regressor = Sequential()

	#Add input and a single hidden layer
	regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
	regressor.add(Dropout(0.2))

	#Add output layer
	regressor.add(Dense(units=1))
	
	regressor.compile(optimizer='adam', loss = 'mean_squared_error')
	regressor.fit(x_train, y_train, epochs = 50, batch_size = 30, validation_split = 0.67)

main()
