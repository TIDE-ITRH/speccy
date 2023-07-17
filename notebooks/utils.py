
import numpy as np

def is_even(x):
    return x % 2 == 0

def n_freq(n):
    return int(np.floor(n/2))

def calc_ff(n, delta = 1):
    return np.arange(n_freq(n)) / n / delta