import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import math

#Read CSV file and transfer into a numpy array
def readData():
	dataset = pd.read_csv('data/Sample.csv',  parse_dates=[1])
	dataset['TIME'] = pd.DatetimeIndex(dataset['TIME']).strftime("%Y-%m-%d %H:%M")
	dataset['monthx'] = (pd.DatetimeIndex(dataset['TIME']).month / 12) * 2 * math.pi
	dataset['monthx'] = dataset['monthx'].apply(math.cos)

	dataset['monthy'] = (pd.DatetimeIndex(dataset['TIME']).month / 12) * 2 * math.pi
	dataset['monthy'] = dataset['monthy'].apply(math.sin)

	dataset['dayx'] = (pd.DatetimeIndex(dataset['TIME']).day / 31) * 2 * math.pi
	dataset['dayx'] = dataset['dayx'].apply(math.cos)

	dataset['dayy'] = (pd.DatetimeIndex(dataset['TIME']).day / 31) * 2 * math.pi
	dataset['dayy'] = dataset['dayy'].apply(math.sin)

	dataset['hourx'] = (pd.DatetimeIndex(dataset['TIME']).hour / 24) * 2 * math.pi
	dataset['hourx'] = dataset['hourx'].apply(math.cos)

	dataset['houry'] = (pd.DatetimeIndex(dataset['TIME']).hour / 24) * 2 * math.pi
	dataset['houry'] = dataset['houry'].apply(math.sin)

	dataset['minutex'] = (pd.DatetimeIndex(dataset['TIME']).minute / 60) * 2 * math.pi
	dataset['minutex'] = dataset['minutex'].apply(math.cos)

	dataset['minutey'] = (pd.DatetimeIndex(dataset['TIME']).minute / 60) * 2 * math.pi
	dataset['minutey'] = dataset['minutey'].apply(math.sin)

	dataset['day_of_weekx'] = (pd.DatetimeIndex(dataset['TIME']).dayofweek / 7) * 2 * math.pi
	dataset['day_of_weekx'] = dataset['day_of_weekx'].apply(math.cos)

	dataset['day_of_weeky'] = (pd.DatetimeIndex(dataset['TIME']).dayofweek / 7) * 2 * math.pi
	dataset['day_of_weeky'] = dataset['day_of_weeky'].apply(math.sin)
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

test = readData()
test = cleanData(test)
np.savetxt('data/polar_load_by_minute.csv', test, delimiter=',')
