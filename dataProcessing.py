import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read CSV file and transfer into a numpy array with valid info
def readData():
	dataset = pd.read_csv('data/Sample.csv')
	dataset_array = dataset.iloc[:, 1:].to_numpy()
	
	#Adjust final column to keep units consistent 
	dataset_array[:, -1] *= 1000

	#All entries with negative numbers in second column are garbage
	dataset_array = dataset_array[dataset_array[..., 1] >= 0]

	#Calculate the total demand of the lab
	dataset_array[:, -1] += dataset_array[:, 1]
	print(dataset_array)
	
	plt.plot(dataset_array[0], dataset_array[1], color = 'red', label = 'Generated Solar Power')
	plt.plot(dataset_array[0], dataset_array[2], color = 'blue', label = 'Lab Electrical Demand')
	plt.show()

readData()
