import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read CSV file and transfer into a numpy array with valid info
def readData():
	dataset = pd.read_csv('data/Sample.csv')
	dataset_array = dataset.iloc[:, 1:].to_numpy()
	plt.plot(dataset_array[0], dataset_array[1:])
	plt.show()

readData()
