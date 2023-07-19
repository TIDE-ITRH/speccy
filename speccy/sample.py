
import numpy as np
import scipy.linalg as linalg
from . import utils as ut
from . import sick_tricks as gary

def mv_gaussian(acf):
    
    n = acf.size
    return np.random.multivariate_normal(np.repeat(0, n), linalg.toeplitz(acf))

def random_amplitudes(specfunc, params, n, delta, alias=False, tol=1e-6):

    ff = ut.fftfreq(n, delta)

    if alias:
        psd_true = ut.fftshift(gary.alias_spectrum(specfunc, params, n, tol))[1::2]
    else:
        psd_true = specfunc(ff, params)
    
    psd_sample = psd_true * np.random.chisquare(2, size=n)

    amp_sample = np.sqrt(psd_sample/n/delta)
    phase_sample = np.random.uniform(0, 2*np.pi, size = n)

    t = ut.taus(n, delta)
    ts_sample = np.repeat(0.0, n)

    for i in np.arange(n):
        f = ff[i]
        ts_sample += amp_sample[i] * np.cos(2*np.pi*f*t - phase_sample[i])

    return ts_sample
