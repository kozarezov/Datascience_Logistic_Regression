#!/usr/bin/env python3
import argparse as arg
import numpy as np
import functions as func
import matplotlib.pyplot as plt
import pandas as pd
from normalize import NormalizeData
from sklearn.metrics import accuracy_score

""" https://www.youtube.com/watch?v=1vklt6IHeJI """

class LogisticRegression:
	def __init__(self, m, n, X_test, y_test):
		# Инициализация w и b случайными числами, report_every - каждые n итераций вывод значения ошибки
		self.n = n
		self.m = m
		self.X_test = X_test
		self.y_test = y_test
		self.w = np.random.randn(n, 1) * 0.001
		self.b = np.random.randn() * 0.001
		self.report_every = 40
	
	# Функция сигмоиды
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	# Функция производных логистических потерь
	def log_loss(self, y_true, y_pred):
		return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=0) / len(y_true)
	
	# Функция предсказания
	def predict(self, X):        
		return np.array([self.sigmoid(x.reshape(1, self.n).dot(self.w) + self.b)[0][0] for x in X])

	def train(self, X, y, learning_rate=0.005, epochs=40):
		""" X - выборка баллов
			y - выборка факультетов
			learning_rate - шаг градиентного спуска
			epochs - количество прохождений по всему датасету  """
		# Заводим 2 массива для отслеживания изменений ошибки
		self.losses_train = [] 
		self.losses_test = []

		for epoch in range(epochs):
			Z = X.reshape(self.m, self.n).dot(self.w) + self.b
			A = self.sigmoid(Z)
			
			dw = np.sum(X.reshape(self.m, self.n) * (A.reshape(self.m, 1) - y.reshape(self.m, 1)), axis=0) / len(X)
			db = np.sum((A.reshape(self.m, 1) - y.reshape(self.m, 1)), axis=0) / len(X)
			
			# Градиентный шаг
			self.w = self.w - learning_rate * dw.reshape(self.n, 1)
			self.b = self.b - learning_rate * db
			
			# Сохраняем ошибку для построение графика
			if epoch % self.report_every == 0:
				self.losses_train.append(self.log_loss(y, self.predict(X)))
				self.losses_test.append(self.log_loss(self.y_test, self.predict(self.X_test)))


# Получаем аргументы
parser = arg.ArgumentParser()
parser.add_argument("dataset", type=str, help="input dataset")
args = parser.parse_args().dataset

def logreg_train(filename):
	data = pd.read_csv(filename) # Читаем датасет
	# Удаляем Nan значения
	data = data.dropna(subset=['Defense Against the Dark Arts'])
	data = data.dropna(subset=['Charms'])
	data = data.dropna(subset=['Herbology'])
	data = data.dropna(subset=['Divination'])
	data = data.dropna(subset=['Muggle Studies'])
	X = np.array(data.values[:, [9, 17, 8, 10, 11]], dtype=float) # Массив баллов по предметам
	y = data.values[:, 1] # Массив факультетов

	normalizer = NormalizeData()
	y = normalizer.binarize(y) # Переводим факультеты в бинарный вид
	X_train, X_test, y_train, y_test = normalizer.train_test_split(X, y, train_size=0.7, random_state=4) # Разбиваем датасет на train и test для оценки качества прогноза
	m, n = X_train.shape # m - количество объектов n - размерность в train выборке
	normalizer.fit(X_train) # собираем данные для нормализации
	X_train_std = normalizer.normolize(X_train) # нормализуем train
	X_test_std = normalizer.normolize(X_test) # нормализуем test

	""" logreg = LogisticRegression(m, n, X_test_std, y_test)
	logreg.train(X_train_std, y_train, epochs=500)

	domain = np.arange(0, len(logreg.losses_train)) * logreg.report_every
	plt.plot(domain, logreg.losses_train, label='Train')
	plt.plot(domain, logreg.losses_test, label='Test')
	plt.xlabel('Epoch number')
	plt.ylabel('LogLoss')
	plt.legend()
	plt.show() """

logreg_train(args)
