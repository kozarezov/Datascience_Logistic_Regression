import numpy as np

def my_count(n):
  try:
    n = n.astype('float')
    n = n[np.logical_not(np.isnan(n))]
    return len(n)
  except:
    return len(n)

def my_mean(n):
  result = 0
  for i in n:
    if np.isnan(i):
      continue
    result = result + i
  return result / len(n)

def my_std(n):
  mean = my_mean(n)
  result = 0
  for i in n:
    if np.isnan(i):
      continue
    result = result + (i - mean) ** 2
  return (result / len(n)) ** 0.5

def my_min(n):
  min_value = n[0]
  for i in n:
    val = i
    if val < min_value:
      min_value = val
  return min_value

def my_max(n):
  min_value = n[0]
  for i in n:
    val = i
    if val > min_value:
      min_value = val
  return min_value

def my_percentile(n, p):
  n.sort()
  k = (len(n) - 1) * (p / 100)
  f = np.floor(k)
  c = np.ceil(k)
  if f == c:
    return n[int(k)]
  d0 = n[int(f)] * (c - k)
  d1 = n[int(c)] * (k - f)
  return d0 + d1