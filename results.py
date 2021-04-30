import neuralNet
import numpy as np
import matplotlib.pyplot as plt

x = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
y = []
y.append(int(neuralNet.evaluateModel('models/version_3', 'data/load_by_minute.csv')))
y.append(int(neuralNet.evaluateModel('models/version_4', 'data/load_by_minute.csv')))
y.append(int(neuralNet.evaluateModel('models/version_5', 'data/load_by_hour.csv')))
y.append(int(neuralNet.evaluateModel('models/version_6', 'data/load_by_hour.csv')))
y.append(int(neuralNet.evaluateModel('models/version_7', 'data/load_by_hour.csv')))
y.append(int(neuralNet.evaluateModel('models/version_8', 'data/load_by_hour.csv')))
y.append(int(neuralNet.evaluateModel('models/version_9', 'data/load_by_minute.csv')))
y.append(int(neuralNet.evaluateModel('models/version_10', 'data/load_by_minute.csv')))
y.append(int(neuralNet.evaluateModel('models/version_11', 'data/polar_load_by_minute.csv')))
y.append(int(neuralNet.evaluateModel('models/version_12', 'data/polar_load_by_minute.csv')))

y = np.array(y)

plt.scatter(x, y)

for i, txt in enumerate(y):
	plt.annotate(txt, (x[i], y[i]))
plt.xlabel('Version Number')
plt.ylabel('Average Error in Watts')
plt.title('Model Results Over Time')
plt.savefig('figures/results.png')
