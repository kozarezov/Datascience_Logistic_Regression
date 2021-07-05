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

# Делим массивы на факультеты
def index_for_array(array, index_column, ind):
	arr = np.array(array[:, index_column], dtype=float)
	Grynffindor = arr[:ind[1]]
	Grynffindor = Grynffindor[np.logical_not(np.isnan(Grynffindor))]
	Hufflepuff = arr[ind[1]:ind[2]]
	Hufflepuff = Hufflepuff[np.logical_not(np.isnan(Hufflepuff))]
	Ravenclaw = arr[ind[2]:ind[3]]
	Ravenclaw = Ravenclaw[np.logical_not(np.isnan(Ravenclaw))]
	Slytherin = arr[ind[3]:]
	Slytherin = Slytherin[np.logical_not(np.isnan(Slytherin))]
	return (Grynffindor, Hufflepuff, Ravenclaw, Slytherin)

# Ищем индекс одногородного распределения баллов
def find_homogeneous(header, data, ind):
	total_diff = []
	for i in range (6, len(header)):
		diff = 0
		Grynffindor, Hufflepuff, Ravenclaw, Slytherin = index_for_array(data, i, ind)
		for facultie in Grynffindor, Hufflepuff, Ravenclaw, Slytherin:
			diff += abs(func.ft_std(facultie))
		total_diff.append(diff)
	index = total_diff.index(func.ft_min(total_diff))
	return (index)

# Рисуем диаграмму
ind = search_index_string(faculties, data)
index = find_homogeneous(header, data, ind)
Grynffindor, Hufflepuff, Ravenclaw, Slytherin = index_for_array(data, index, ind)
plt.hist(Grynffindor, color='red', alpha=0.5)
plt.hist(Hufflepuff, color='yellow', alpha=0.5)
plt.hist(Ravenclaw, color='blue', alpha=0.5)
plt.hist(Slytherin, color='green', alpha=0.5)
plt.legend(faculties, loc='upper right', frameon=True)
plt.title(header[index])
plt.xlabel('Score')
plt.ylabel('Students count')

plt.show()
