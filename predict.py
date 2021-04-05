import dataProcessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def getData():
	test = dataProcessing.readData()
	test = dataProcessing.cleanData(test)
	
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

	return x_train, y_train, x_test, y_test, sc

	
def trainModel(modelName):

	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense
	from tensorflow.keras.layers import LSTM 
	from tensorflow.keras.layers import Dropout

	x_train, y_train, _, _, _ = getData()

	#Create model for learning
	regressor = Sequential()

	#Add input and a single hidden layer
	regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
	regressor.add(Dropout(0.2))

	#Add output layer
	regressor.add(Dense(units=1))
	
	regressor.compile(optimizer='adam', loss = 'mean_squared_error')
	regressor.fit(x_train, y_train, epochs = 100, batch_size = 50)

	regressor.save(modelName)

def predict(modelName):
	from tensorflow import keras
	model = keras.models.load_model(modelName)
	_, _, x_test, y_test, sc = getData()
	predicted_results = model.predict(x_test)

	import matplotlib.pyplot as plt
	plt.plot(y_test, color = 'red', label = "Actual Load")
	plt.plot(predicted_results[:, 0], color = 'blue', label = 'Predicted Load')
	plt.title("Electrical Load Prediction")
	plt.ylabel("Demand in Watts")
	plt.legend()
	plt.savefig("figures/version_0_results.png")

	
