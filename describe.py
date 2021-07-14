#!/usr/bin/env python3
import argparse as arg
import numpy as np
import functions as func
from colorama import init, Fore, Back, Style

# Включаем сбор цваета после печати
init(autoreset=True)

# Получаем аргументы
parser = arg.ArgumentParser()
parser.add_argument("dataset", type=str, help="input dataset")
args = parser.parse_args().dataset

def add_header_column():
	column = []
	column.append(f'{"":15}')
	column.append(f'{"Count":>12}')
	column.append(f'{"Mean":>12}')
	column.append(f'{"Std":>12}')
	column.append(f'{"Min":>12}')
	column.append(f'{"25%":>12}')
	column.append(f'{"50%":>12}')
	column.append(f'{"75%":>12}')
	column.append(f'{"Max":>12}')
	return column

def print_value(data, i):
	try:
		arr = np.array(data[:, i], dtype=float)
		arr = arr[np.logical_not(np.isnan(arr))]
		if not arr.any():
			raise Exception()
		count_value = func.ft_count(arr)
		print(f'{count_value:>12.2f}', end=' |')
		mean_value = func.ft_mean(arr)
		print(f'{Fore.RED if mean_value < 0 else Fore.RESET}' + f'{mean_value:>12.2f}', end=' |')
		std_value = func.ft_std(arr)
		print(f'{Fore.RED if std_value < 0 else Fore.RESET}' + f'{std_value:>12.2f}', end=' |')
		min_value = func.ft_min(arr)
		print(f'{Fore.RED if min_value < 0 else Fore.RESET}' + f'{min_value:>12.2f}', end=' |')
		perc25 = func.ft_percentile(arr, 25)
		print(f'{Fore.RED if perc25 < 0 else Fore.RESET}' + f'{perc25:>12.2f}', end=' |')
		perc50 = func.ft_percentile(arr, 50)
		print(f'{Fore.RED if perc50 < 0 else Fore.RESET}' + f'{perc50:>12.2f}', end=' |')
		perc75 = func.ft_percentile(arr, 75)
		print(f'{Fore.RED if perc75 < 0 else Fore.RESET}' + f'{perc75:>12.2f}', end=' |')
		max_value = func.ft_max(arr)
		print(f'{Fore.RED if max_value < 0 else Fore.RESET}' + f'{max_value:>12.2f}')
	except:
		count_value = func.ft_count(data[:, i])
		print(f'{count_value:>12.2f}', end=' |')
		print(f'{" - ":>12}', end=' |')
		print(f'{" - ":>12}', end=' |')
		print(f'{" - ":>12}', end=' |')
		print(f'{" - ":>12}', end=' |')
		print(f'{" - ":>12}', end=' |')
		print(f'{" - ":>12}', end=' |')
		print(f'{" - ":>12}')

def describe(filename):
	data = func.read_csv(filename)
	header_string = data[0]
	data = data[1:, :]
	header_column = add_header_column()
	for i in range(0, len(header_column) - 1):
		print(Fore.GREEN + f'{header_column[i]:>12}', end = ' |')
	print(Fore.GREEN + f'{header_column[i + 1]:>12}')
	for i in range(0, len(header_string)):
		print(Fore.CYAN + f'{header_string[i]:15.15}', end = ' |')
		print_value(data, i)
	

describe(args)