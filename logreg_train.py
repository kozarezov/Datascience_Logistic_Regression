#!/usr/bin/env python3
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from normalize import NormalizeData
from logreg_predict import Predict
from sklearn.metrics import accuracy_score

OUT = './weights.csv'

class LogisticRegression:
	def __init__(self, learning_rate=0.1, max_iterations=400):
		# Инициализация w и b случайными числами, report_every - каждые n итераций вывод значения ошибки
		self.lr = learning_rate
		self.iterations = max_iterations
		self.report_every = 10 # каждые n итераций фиксировать данные для построения графика
	
	def gradient_descent(self, X, theta, Y, m):
		Z = X.dot(theta)
		A = NormalizeData.sigmoid(Z)
		gradient = np.dot(X.T, (A - Y)) / m
		return self.lr * gradient

	# Функция рассчета логистических потерь
	def log_loss(self, y_true, y_pred):
		return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=0) / len(y_true)

	def one_vs_all(self, Y, it):
		y_binar = np.zeros(len(Y), dtype=int)
		for i, y in enumerate(Y):
			if (it == y):
				y_binar[i] = 1
		return y_binar

	def train(self, X, Y):
		m, n = X.shape # m - количество объектов n - размерность в train выборке
		self.theta = []
		self.losses_train = []
		theta_nb = len(X[0])

		for it in range(4):
			losses_train = []
			y_binar = self.one_vs_all(Y, it)
			theta = np.zeros(theta_nb)
			for epoch in range(self.iterations):
				theta -= self.gradient_descent(X, theta, y_binar, m)
				if epoch % self.report_every == 0:
					losses_train.append(self.log_loss(y_binar, Predict.predict_for_plot(X, theta)))
			self.losses_train.insert(0, losses_train)
			self.theta.append(theta)

def logreg_train(filename):
	data = pd.read_csv(filename) # Читаем датасет
	data = data.dropna()
	courses = data.columns[6:].values.tolist()
	X = np.array(data.values[:, 6:], dtype=float)
	""" (Herbology,Defense Against the Dark Arts,Divination,Ancient Runes,Flying)
	courses = data.columns[[8, 9, 10, 12, 18]].values.tolist()
	X = np.array(data.values[:, [8, 9, 10, 12, 18]], dtype=float) """
	y = data.values[:, 1] # Массив факультетов
	
	normalizer = NormalizeData()
	k, y = normalizer.numberize(y) # Переводим факультеты в цифровой вид
	X_train, X_test, y_train, y_test = normalizer.train_test_split(X, y, train_size=0.7, random_state=4) # Разбиваем датасет на train и test для оценки качества прогноза
	normalizer.fit(X_train) # собираем данные для нормализации
	X_train_std = normalizer.normolize(X_train) # нормализуем train
	X_test_std = normalizer.normolize(X_test) # нормализуем test

	logreg = LogisticRegression(learning_rate=0.1, max_iterations=400)
	logreg.train(X_train_std, y_train) # обучаем модель на тренировочной выборке
	logpred = Predict()
	y_predict = logpred.predict(X_test_std, logreg.theta) # прогнозируем тестовую выборку
	
	accuracy = accuracy_score(y_test, y_predict) # проверяем точность
	print(f"Точность: {round(accuracy * 100, 2)} %")

	# Выводим потери на график
	domain = np.arange(0, len(logreg.losses_train[0])) * logreg.report_every
	for i in range(4):
		plt.plot(domain, logreg.losses_train[i], label=k[i])
	plt.xlabel('Iterations')
	plt.ylabel('LogLoss')
	plt.legend()
	plt.show()

	# Записываем в файл
	with open(OUT, 'w') as res:
		writer = csv.writer(res)
		writer.writerow(courses)
		for theta in logreg.theta:
			writer.writerow(theta)
		writer.writerow(normalizer.mean)
		writer.writerow(normalizer.max_min)
		
# Получаем аргументы
parser = arg.ArgumentParser()
parser.add_argument("dataset", type=str, help="укажите путь к dataset")
args = parser.parse_args().dataset
logreg_train(args)
