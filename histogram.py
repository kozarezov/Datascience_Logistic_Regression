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

# Ищем индекс одногородного распределения баллов
def find_homogeneous(header, data, ind):
	total_diff = []
	for i in range (6, len(header)):
		diff = 0
		Grynffindor, Hufflepuff, Ravenclaw, Slytherin = func.index_for_array(data, i, ind)
		for facultie in Grynffindor, Hufflepuff, Ravenclaw, Slytherin:
			diff += abs(func.ft_std(facultie))
		total_diff.append(diff)
	index = total_diff.index(func.ft_min(total_diff)) + 6
	return (index)

# Рисуем диаграмму
ind = func.search_index_string(faculties, data)
index = find_homogeneous(header, data, ind)
Grynffindor, Hufflepuff, Ravenclaw, Slytherin = func.index_for_array(data, index, ind)
plt.hist(Grynffindor, color='red', alpha=0.5)
plt.hist(Hufflepuff, color='yellow', alpha=0.5)
plt.hist(Ravenclaw, color='blue', alpha=0.5)
plt.hist(Slytherin, color='green', alpha=0.5)
plt.legend(faculties, loc='upper right', frameon=True)
plt.title(header[index])
plt.xlabel('Баллы')
plt.ylabel('Количество студентов')

plt.show()
