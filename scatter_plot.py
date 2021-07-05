import functions as func
import numpy as np
import matplotlib.pyplot as plt

PATH = './datasets/dataset_train.csv'

# Читаем и сортируем массив
data = func.read_csv(PATH)
header = data[0]
data = data[1:, :]
data = data[data[:, 1].argsort()]
faculties = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

# Находим индексы факультетов по строкам
def search_index_string(faculties, data):
	ind = [0]
	for it in faculties:
		index = np.searchsorted(data[:, 1], it)
		ind.append(index)
	ind = np.unique(ind)
	return (ind)

# Нормализуем массив для нахождения схожих значений
def normalize(data, index):
	arr = np.array(data[:, index], dtype=float)
	arr = arr[~np.isnan(arr)]
	arr = [abs(grade) for grade in arr]
	arr = [int(str(grade).replace('.', '')[:10]) for grade in arr]
	return arr

# Находим схожие значения
def find_similar(header, data):
	total_diff = []
	for idx in range (6, len(header)):
		header_1 = normalize(data, idx)
		for idx2 in range(idx + 1, len(header)):
			header_2 = normalize(data, idx2)
			diff = set(header_1) - set(header_2)
			length = (len(header_1) + len(header_2)) // 2
			five_percent = (length * 5) // 100
			if len(diff) <= five_percent:
				total_diff.append(idx)
				total_diff.append(idx2)
	return total_diff


# Рисуем диаграмму
ind = search_index_string(faculties, data)
index = find_similar(header, data)

x = np.array(data[:, index[0]], dtype=float)
y = np.array(data[:, index[1]], dtype=float)

plt.scatter(x[:ind[1]], y[:ind[1]], color='red', alpha=0.5)
plt.scatter(x[ind[1]:ind[2]], y[ind[1]:ind[2]], color='yellow', alpha=0.5)
plt.scatter(x[ind[2]:ind[3]], y[ind[2]:ind[3]], color='blue', alpha=0.5)
plt.scatter(x[ind[3]:], y[ind[3]:], color='green', alpha=0.5)
plt.legend(faculties, loc='upper right', frameon=False)
plt.xlabel(header[7])
plt.ylabel(header[9])
plt.show()