import numpy as np
import utils as ut

def bochner(acf, delta = 1, alias = True, bias = True):

    n = np.size(acf)

    if bias:
        acf = (1 - np.arange(n)/n) * acf

    ff = ut.calc_ff(n, delta)

    if ut.is_even(n):
        acf = np.concatenate([[acf[0]/2], acf[1:(n-1)], [acf[-1]/2]])
    else:
        acf = np.concatenate([[acf[0]/2], acf[1:n]])
    
    psd = 2 * delta * np.real(np.fft.fft(acf))[:ut.n_freq(n)]

    return ff, psd

def inv_bochner(myfunc, params, n, delta = 1, alias = True):
    
    ff1 = np.arange(1, n) / (2*n)

    S_zero = myfunc(0, params)
    S_ff = myfunc(ff1, params)
    S_nyq = myfunc(0.5, params)

    S = np.concatenate([[S_zero], S_ff, [S_nyq], S_ff[::-1]])

    return np.real(np.fft.ifft(S))