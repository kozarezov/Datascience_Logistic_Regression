import numpy as np
import functions as func

class NormalizeData(object):
	def __init__(self, mean=np.array([]), std=np.array([])):
		self._mean = mean
		self._std = std

	def fit(self, X):
		for i in range(0, X.shape[1]):
			self._mean = np.append(self._mean, func.ft_mean(X[:, i]))
			self._std = np.append(self._std, func.ft_std(X[:, i]))

	def normolize(self, X):
		return ((X - self._mean) / self._std)
	
	# Аналог LabelBinarizer из sklearn
	def binarize(self, Y):
		k = np.unique(Y).tolist()
		y = np.zeros((len(Y), len(k)), dtype=int)
		for i in range(0, len(Y)):
			y[i, k.index(Y[i])] = 1
		return y

	# Аналог train_test_split из sklearn
	def train_test_split(self, X, y, train_size=0.7, random_state=None):
		if random_state:
			np.random.seed(random_state)
		p = np.random.permutation(len(X))

		X_offset = int(len(X) * train_size)
		y_offset = int(len(y) * train_size)

		X_test = X[p][X_offset:]
		X_train = X[p][:X_offset]
		
		y_test = y[p][y_offset:]
		y_train = y[p][:y_offset]
		return (X_train, X_test, y_train, y_test)