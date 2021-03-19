import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

#Read CSV file and transfer into a numpy array
def readData():
	dataset = pd.read_csv('data/Sample.csv',  parse_dates=[1])
	dataset['month'] = pd.DatetimeIndex(dataset['TIME']).month
	dataset['day'] = pd.DatetimeIndex(dataset['TIME']).day
	dataset['hour'] = pd.DatetimeIndex(dataset['TIME']).hour
	dataset['minute'] = pd.DatetimeIndex(dataset['TIME']).minute
	dataset['second'] = pd.DatetimeIndex(dataset['TIME']).second
	dataset['day_of_week'] = pd.DatetimeIndex(dataset['TIME']).dayofweek
	dataset_array = dataset.iloc[:, 2:].to_numpy()
	return dataset_array

#Clean data by removing grabage values and normalizing units
def cleanData(dataset_array):
	#Adjust final column to keep units consistent 
	dataset_array[:, 1] *= 1000

	#All entries with negative numbers in second column are garbage
	dataset_array = dataset_array[dataset_array[..., 0] >= 0]

	#Calculate the total demand of the lab
	dataset_array[:, 1] += dataset_array[:, 0]
	return dataset_array[:, 1:]

	'''
	plt.plot(dataset_array[0], dataset_array[1], color = 'red', label = 'Generated Solar Power')
	plt.plot(dataset_array[0], dataset_array[2], color = 'blue', label = 'Lab Electrical Demand')
	plt.show()
	'''

test = readData()
test = cleanData(test)
print(test[0])
