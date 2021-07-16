import numpy as np
import functions as func
from normalize import NormalizeData

class Predict():
	def predict(self, X, all_thetas):
		X = np.insert(X, 0, 1, axis=1)
		Y = []
		for x in X:
			probability_of_y = []
			for y_thetas in all_thetas:
				z = np.dot(y_thetas.T, x)
				a = NormalizeData.sigmoid(z)
				probability_of_y.append(a)
			y_max = func.ft_max(probability_of_y)
			Y.append(probability_of_y.index(y_max))
		return Y
	
	def predict_for_plot(X, theta):
		return np.array([NormalizeData.sigmoid(x.dot(theta)) for x in X])