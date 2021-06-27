import argparse as arg
import numpy as np
import csv
import functions as func

# Получаем аргумент
parser = arg.ArgumentParser()
parser.add_argument("dataset", type=str, help="input dataset")
args = parser.parse_args().dataset

# Функция чтения из файла
def read_csv(filename):
	data = list()
	with open(filename) as csv_file:
		input = csv.reader(csv_file)
		try:
			for i in input:
				row = list()
				for value in i:
					try:
						value = float(value)
					except:
						if not value:
							value = np.nan
					row.append(value)
				data.append(row)
		except csv.Error as e:
			exit(e)
	return np.array(data, dtype=object)

def add_header_column():
	column = []
	column.append('|              |')
	column.append('|    Count     |')
	column.append('|    Mean      |')
	column.append('|    Std       |')
	column.append('|    Min       |')
	column.append('|    25%       |')
	column.append('|    50%       |')
	column.append('|    75%       |')
	column.append('|    Max       |')
	return column

def add_value_column(dataset, len):
	value_column = list()
	for i in range(0, len):
		try:
			data = np.array(dataset[:, i], dtype=float)
			data = data[~np.isnan(data)]
			if not data.any():
				raise Exception()
			value_column[i:0] = func.my_count(data)
			value_column[i:1] = func.my_mean(data)
			value_column[i:2] = func.my_std(data)
			value_column[i:3] = func.my_min(data)
			value_column[i:4] = func.my_percentile(data, 25)
			value_column[i:5] = func.my_percentile(data, 50)
			value_column[i:6] = func.my_percentile(data, 75)
			value_column[i:7] = func.my_max(data)
		except:
			value_column[i:0] = func.my_count(dataset[:, i])
			value_column[i:1] = " - "
			value_column[i:2] = " - "
			value_column[i:3] = " - "
			value_column[i:4] = " - "
			value_column[i:5] = " - "
			value_column[i:6] = " - "
			value_column[i:7] = " - "
	return (value_column)


def describe(filename):
	data = read_csv(filename)
	header_string = data[0]
	header_column = add_header_column()
	value_column = add_value_column(data[1 : , :], len(header_string))
	print(value_column)

	""" func.my_count(arr)
	func.my_mean(arr)
	func.my_std(arr)
	func.my_min(arr)
	func.my_percentile(arr, 25)
	func.my_percentile(arr, 50)
	func.my_percentile(arr, 75)
	func.my_max(arr) """
	

describe(args)