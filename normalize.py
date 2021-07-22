import numpy as np
import functions as func

class NormalizeData(object):
	def __init__(self):
		self.min = np.array([])
		self.max_min = np.array([])

	def fit(self, X):
		for i in range(0, X.shape[1]):
			self.min = np.append(self.min, func.ft_min(X[:, i]))
			self.max_min = np.append(self.max_min, (func.ft_max(X[:, i] - func.ft_min(X[:, i]))))
	
	def set_fit(self, min, max_min):
		self.min = min
		self.max_min = max_min

	def normolize(self, X):
		return ((X - self.min) / self.max_min)
	
	# Аналог LabelBinarizer из sklearn
	def numberize(self, Y):
		k = np.unique(Y).tolist()
		y = np.zeros(len(Y), dtype=int)
		for i in range(0, len(Y)):
			y[i] = k.index(Y[i])
		return k, y

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

	# Функция сигмоиды
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))