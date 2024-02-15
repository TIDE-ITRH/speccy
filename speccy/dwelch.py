
import numpy as np
from scipy.signal import welch
from . import utils as ut
from . import sick_tricks as st

def get_centres(l, k, delta = 1):
    
    if ut.is_even(l):
        width = 0.5/(k+1)
    else:
        width = 0.5/(k+0.5)
        
    centres = width * np.arange(1, k+1) / delta

    return centres, np.repeat(width,k)

def sinc(x):
    value = np.sin(x)/x
    value[0] = 1
    return value

def pwelch(ts, m, l, s=None, delta=1, h=None, overlap=None):
    
    n = (m - 1) * s + l
    nfreq = ut.n_freq(l)

    # Calculate symmetric spectrum, shift to start at zero, and take points inbetween zero and Nyquist
    ff = ut.fftshift(ut.fftfreq(l, delta=1))[1:(nfreq+1)]
    pxx = np.empty((nfreq, m))

    if h is None:
        h = np.repeat(1, l)

    h = h/np.sqrt(np.sum(h**2))

    for i in np.arange(1,(m+1)):
        start = (i - 1) * s
        end = start + l
        ts_tmp = ts[start:end]
        _, S = st.periodogram(ts_tmp, h=h, return_onesided=True)
        pxx[:,i-1] = S[1:nfreq+1]

    return ff, np.mean(pxx, 1)

def build_bases(l, k = None, h = None, delta = 1):

    nfreq = ut.n_freq(l)
    centres, widths = get_centres(l, k, delta)
    
    bases = np.empty((nfreq, k))
    tt = np.arange(l)

    for i in np.arange(k):
        acf_tmp = 2 * widths[i] * sinc(np.pi * tt * widths[i]) * np.cos(2 * np.pi * centres[i] * tt)
        _, S = st.bochner(acf_tmp, delta=1, bias=True, h=h, return_onesided=True)
        bases[:,i] = S[1:]

    return bases

def dwelch(ts, m, l, s, k = None, delta = 1, h = None, model = 'vanilla'):

    _, pw = pwelch(ts, m, l, s, h=h)
    
    L = np.diag(1/pw)
    b = L @ pw

    A = L @ build_bases(l, k=k, h=h, delta=delta)
    centres, _ = get_centres(l, k, delta=delta)

    if model == 'vanilla':
        dw = np.linalg.inv(A.T @ A) @ A.T @ b

    return centres, dw