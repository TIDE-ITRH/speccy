
import numpy as np
import scipy.linalg as linalg
import utils as ut
import sick_tricks as gary

def mv_gaussian(acf):
    
    n = acf.size
    return np.random.multivariate_normal(np.repeat(0, n), linalg.toeplitz(acf))

def random_amplitudes(specfunc, params, n, delta, alias=False, tol=1e-6):
    
    ff_pos = ut.calc_ff(n, delta)[1:]
    ff_neg = -ff_pos[::-1]
    ff_concat = np.concatenate([ff_neg, [0], ff_pos, [0.5/delta]])

    if alias:
        psd_true = gary.alias_spectrum(specfunc, params, n, tol)
        psd_zero = psd_true[0]
        psd_pos = psd_true[1:n][2::2]
        psd_nyq = psd_true[n]
        psd_neg = psd_true[(n+1):][2::2]
        psd_true = np.concatenate([psd_neg, [psd_zero], psd_pos, [psd_nyq]])

    else:
        psd_neg = specfunc(ff_neg, params)
        psd_pos = specfunc(ff_pos, params)
        psd_zero = [0]
        psd_nyq = [specfunc(0.5, params)]
        psd_true = np.concatenate([psd_neg, psd_zero , psd_pos, psd_nyq])
    
    psd_sample = psd_true * np.random.chisquare(2, size=n)

    amp_sample = np.sqrt(psd_sample/n/delta)
    phase_sample = np.random.uniform(0, 2*np.pi, size = n)

    t = np.arange(0, n) * delta
    ts_sample = np.repeat(0.0, n)

    for i in np.arange(n):
        f = ff_concat[i]
        ts_sample += amp_sample[i] * np.cos(2*np.pi*f*t - phase_sample[i])

    return ts_sample
