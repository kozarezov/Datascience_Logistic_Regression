import numpy as np
import functions as func
from normalize import NormalizeData

class Predict():
	def h(self, X, thetas):
		z = np.dot(thetas.T, X)
		return NormalizeData.sigmoid(z)

	def predict(self, X, all_thetas):
		X = np.insert(X, 0, 1, axis=1)
		Y = []
		for x in X:
			probability_of_y = []
			for y_thetas in all_thetas:
				h = self.h(x, y_thetas)
				probability_of_y.append(h)
			y_max = func.ft_max(probability_of_y)
			Y.append(probability_of_y.index(y_max))
		return Y