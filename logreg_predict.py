#!/usr/bin/env python3
import numpy as np
import functions as func
import argparse as arg
import pandas as pd
import csv
from normalize import NormalizeData

class Predict():
	def predict(self, X, all_thetas):
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

def open_file(dataset, weights):
	# Читаем файл с результатами
	with open(weights) as weight:
		data = csv.reader(weight)
		k = next(data)
		thetas = []
		for _ in range(4):
			theta = np.array(next(data)).astype(float)
			thetas.append(theta)
	
		min = np.array(next(data)).astype(float)
		max = np.array(next(data)).astype(float)

	# Читаем файл с данными для прогноза
	data = pd.read_csv(dataset)
	data = data.fillna(0)
	X = np.array(data.values[:, 6:], dtype=float)

	#Нормализуем данные
	normalizer = NormalizeData()
	normalizer.set_fit(min, max)
	X_std = normalizer.normolize(X)

	return thetas, X_std

def logreg_predict(dataset, weights):
	thetas, X = open_file(dataset, weights)
	faculties = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

	logpred = Predict()
	y_predict = logpred.predict(X, thetas)

	with open("houses.csv", 'w') as res:
		writer = csv.writer(res)
		writer.writerow(["Index", "Hogwarts House"])
		for i, y in enumerate(y_predict):
			writer.writerow([i, faculties[y]])

# Получаем аргументы
if __name__ == '__main__':
	parser = arg.ArgumentParser()
	parser.add_argument("dataset", type=str, help="укажите путь к dataset")
	parser.add_argument("weights", type=str, help="укажите путь к weights")
	args = parser.parse_args()
	logreg_predict(args.dataset, args.weights)