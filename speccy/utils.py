
import numpy as np

def is_even(x):
    return x % 2 == 0

def n_freq(n):
    """Number of positive frequencies excluding Nyquist"""
    
    if is_even(n):
        return int(np.floor(n/2)) - 1
    else:
        return int(np.floor(n/2))

def taus(n, delta=1):
    return delta * np.arange(n)

def fftshift(x):
    return np.fft.fftshift(x)

def fftfreq(n, delta=1):
    return fftshift(np.fft.fftfreq(n, delta))