import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

#Read CSV file and transfer into a numpy array
def readData():
	dataset = pd.read_csv('data/Sample.csv',  parse_dates=[1])
	dataset['TIME'] = pd.DatetimeIndex(dataset['TIME']).strftime("%Y-%m-%d %H:%M")
	dataset['month'] = pd.DatetimeIndex(dataset['TIME']).month
	dataset['day'] = pd.DatetimeIndex(dataset['TIME']).day
	dataset['hour'] = pd.DatetimeIndex(dataset['TIME']).hour
	dataset['minute'] = pd.DatetimeIndex(dataset['TIME']).minute
	dataset['day_of_week'] = pd.DatetimeIndex(dataset['TIME']).dayofweek
	dataset = dataset.groupby(['TIME']).max()
	dataset_array = dataset.iloc[:, 1:].to_numpy()
	return dataset_array

#Clean data by removing grabage values and normalizing units
def cleanData(dataset_array):
	#Adjust final column to keep units consistent 
	dataset_array[:, 1] *= 1000

	#All entries with negative numbers in second column are garbage
	dataset_array = dataset_array[dataset_array[..., 0] >= 0]

	#Calculate the total demand of the lab
	dataset_array[:, 1] += dataset_array[:, 0]
	dataset_array = dataset_array[dataset_array[..., 1] >= 0]
	return dataset_array[:, 1:]

"""
test = readData()
test = cleanData(test)
plt.plot(test[:, 0], color="blue")
plt.title("Electrical Demand Over Time")
plt.ylabel('Watts')
plt.savefig("figures/test.png")
"""
