import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

def getData(dataset):
	test = pd.read_csv(dataset).to_numpy()
	maxLoad = np.max(test[:, 0])
	
	sc = MinMaxScaler(feature_range = (0, 1))
	allScaled = sc.fit_transform(test)
	trainingSetScaled = test
	trainingSetScaled[:, 0] = allScaled[:, 0]

	x_train = []
	y_train = []
	for i in range(60, len(test)):
		x_train.append(trainingSetScaled[i - 60:i, :])
		y_train.append(trainingSetScaled[i, 0])
	x_train, y_train = np.array(x_train), np.array(y_train)

	trainThreshold = len(test) // 100 * 67
	x_test = x_train[trainThreshold:, :]
	y_test = y_train[trainThreshold:]

	x_train = x_train[:trainThreshold, :]
	y_train = y_train[:trainThreshold]

	return x_train, y_train, x_test, y_test, maxLoad 

	
def trainModel(modelName, dataset):

	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense
	from tensorflow.keras.layers import LSTM 
	from tensorflow.keras.layers import Dropout

	x_train, y_train, _, _, _ = getData(dataset)

	#Create model for learning
	regressor = Sequential()

	#Add input and a single hidden layer
	regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
	regressor.add(Dropout(0.2))

	regressor.add(LSTM(units = 50, return_sequences = True))
	regressor.add(Dropout(0.2))

	regressor.add(LSTM(units = 50, return_sequences = True))
	regressor.add(Dropout(0.2))

	regressor.add(LSTM(units = 50, return_sequences = True))
	regressor.add(Dropout(0.2))

	regressor.add(LSTM(units = 50, return_sequences = False))
	regressor.add(Dropout(0.2))

	#Add output layer
	regressor.add(Dense(units=1, activation='sigmoid'))
	
	regressor.compile(optimizer='adam', loss = 'mean_absolute_error')
	regressor.fit(x_train, y_train, epochs = 50, batch_size = 55)

	regressor.save(modelName)


def evaluateModel(modelName, dataset):
	model = keras.models.load_model(modelName)
	_, _, x_test, y_test, maxLoad = getData(dataset)
	#results = model.evaluate(x_test, y_test)
	results = model.evaluate(x_test, y_test, batch_size = 55)
	print('Raw Loss: ' + str(results))
	print('Scaled Loss: ' + str(results * maxLoad) + ' watts')
	return results * maxLoad


def visualizeModelResults(modelName, dataset):
	model = keras.models.load_model(modelName)
	_, _, x_test, y_test, maxLoad = getData(dataset)
	predicted_results = model.predict(x_test)
	predicted_results *= maxLoad
	y_test *= maxLoad

	version = modelName[7:].title().replace('_', ' ')
	fileName = 'figures' + modelName[6:]

	plt.plot(y_test, color = 'red', label = "Actual Load")
	plt.plot(predicted_results, color = 'blue', label = "Predicted Load")
	plt.ylabel("Demand in Watts")
	plt.legend()
	plt.title(version + ' Load Prediction')
	plt.savefig(fileName + '_time.png')

	plt.clf()

	plt.scatter(predicted_results, y_test)
	plt.ylabel("Actual Demand")
	plt.xlabel("Predicted Demand")
	plt.title(version + ' Prediction Results')
	plt.savefig(fileName + '_scatterplot.png')
