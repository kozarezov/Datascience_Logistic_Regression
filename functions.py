import numpy as np
import csv

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

# Переписанные математические функции
def ft_count(n):
  try:
    n = n.astype('float')
    n = n[np.logical_not(np.isnan(n))]
    return len(n)
  except:
    return len(n)

def ft_mean(n):
  result = 0
  for i in n:
    if np.isnan(i):
      continue
    result = result + i
  return result / len(n)

def ft_std(n):
  mean = ft_mean(n)
  result = 0
  for i in n:
    if np.isnan(i):
      continue
    result = result + (i - mean) ** 2
  return (result / len(n)) ** 0.5

def ft_min(n):
  min_value = n[0]
  for i in n:
    val = i
    if val < min_value:
      min_value = val
  return min_value

def ft_max(n):
  min_value = n[0]
  for i in n:
    val = i
    if val > min_value:
      min_value = val
  return min_value

def ft_percentile(n, p):
  n.sort()
  k = (len(n) - 1) * (p / 100)
  f = np.floor(k)
  c = np.ceil(k)
  if f == c:
    return n[int(k)]
  d0 = n[int(f)] * (c - k)
  d1 = n[int(c)] * (k - f)
  return d0 + d1