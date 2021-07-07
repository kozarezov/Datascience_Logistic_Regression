#!/usr/bin/env python3
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

# Функция для добавления гистограммы
def add_histogram(axes, data, column, ind):
	Grynffindor, Hufflepuff, Ravenclaw, Slytherin = func.index_for_array(data, column, ind)
	axes.hist(Grynffindor, color='red', alpha=0.5)
	axes.hist(Hufflepuff, color='yellow', alpha=0.5)
	axes.hist(Ravenclaw, color='blue', alpha=0.5)
	axes.hist(Slytherin, color='green', alpha=0.5)
	axes.set_yticks([])
	axes.set_xticks([])
 
# Функция для добавления скаттер-плота
def add_scatter_plot(axes, x, y, ind):
	axes.scatter(x[:ind[1]], y[:ind[1]], s=1, color='red', alpha=0.5)
	axes.scatter(x[ind[1]:ind[2]], y[ind[1]:ind[2]], s=1, color='yellow', alpha=0.5)
	axes.scatter(x[ind[2]:ind[3]], y[ind[2]:ind[3]], s=1, color='blue', alpha=0.5)
	axes.scatter(x[ind[3]:], y[ind[3]:], color='green', s=1, alpha=0.5)
	axes.set_yticks([])
	axes.set_xticks([])

def cut_label(header, cut_len):
	return (header[:cut_len] + "\n" + header[cut_len:])

# Функция для добавления названия курса
def add_label(axes, row, column, header):
	if axes[row, column].get_subplotspec().is_last_row():
		axes[row, column].set_xlabel(cut_label(header[column + 6], 16), rotation=30, ha='right', fontsize=8)

	if axes[row, column].get_subplotspec().is_first_col():
		axes[row, column].set_ylabel(cut_label(header[row + 6], 16), rotation=30, ha='right', fontsize=8)

	axes[row, column].spines['right'].set_visible(False)
	axes[row, column].spines['top'].set_visible(False)

# Рисуем общий график
len_header = len(header)
ind = func.search_index_string(faculties, data)
figure, axes = plt.subplots(len_header - 6, len_header - 6, figsize=(14, 7))
for row in range(6, len_header):
	for column in range(6, len_header):
		x = np.array(data[:, column], dtype=float)
		y = np.array(data[:, row], dtype=float)
		if row == column:
			add_histogram(axes[row - 6, column - 6], data, column, ind)
		else:
			add_scatter_plot(axes[row - 6, column - 6], x, y, ind)
		add_label(axes, row - 6, column - 6, header)
figure.legend(faculties, loc='upper center', frameon=True, ncol=4, fontsize=9)
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=1, hspace=0, wspace=0)
plt.show()