
import numpy as np
import utils as ut

def periodogram(ts, delta = 1, h = None):
    
    n = ts.size

    if h is not None:
        norm = np.sum(h**2)
        scale = np.sqrt(n/norm)
        ts = scale * h * ts

    dft = np.fft.fft(ts)/np.sqrt(n/delta)
    
    I = np.real(dft * np.conj(dft))
    ff = np.fft.fftfreq(n, delta)

    return ut.fftshift(ff), ut.fftshift(I)

def whittle(ts, specfunc, params, delta = 1, h = None):
    
    ff, I = periodogram(ts, delta, h)
    S = specfunc(ff, params)

    ll = - (np.log(S) + I/S)
    idx = (ff != 0) * (ff != -0.5/delta)
    
    return np.sum(ll[idx])

def dwhittle(ts, acffunc, params, delta = 1, h = None):
    
    tt = delta * np.arange(ts.size)
    ff, I = periodogram(ts, delta, h)
    ff_boch, S_boch = bochner(acffunc(tt, params), delta = delta, bias = True)
    # HACK: quick fix cause bochner isn't two sided yet
    return - 2 * np.sum(np.log(S_boch[ff_boch > 0]) + I[ff > 0]/S_boch[ff_boch > 0])

def bochner(acf, delta = 1, bias = True, h = None):

    n = np.size(acf)

    if h is not None:
        
        norm = np.sum(h**2)
        h_conv = (np.convolve(h, h, mode = 'full')/norm)[(n-1):]
        acf = h_conv * acf

    elif bias:

        acf = (1 - np.arange(n)/n) * acf

    ff = ut.fftfreq(n, delta)

    if ut.is_even(n):
        acf = np.concatenate([[acf[0]/2], acf[1:(n-1)], [acf[-1]/2]])
    else:
        acf = np.concatenate([[acf[0]/2], acf[1:n]])
    
    psd = 2 * delta * np.real(np.fft.fft(acf))

    return ff, ut.fftshift(psd)

def inv_bochner(myfunc, params, n, delta = 1, alias = False, tol = 1e-6):

    if alias:
        S = alias_spectrum(myfunc, params, n, delta, tol)
    else:
        ff = ut.fftshift(ut.fftfreq(2*n, delta))
        S = myfunc(ff, params)

    acf = np.real(np.fft.ifft(S)) / delta

    return ut.taus(n, delta), acf[:n]

def alias_spectrum(myfunc, params, n, delta, tol = 1e-6):
    
    ff = ut.fftshift(ut.fftfreq(2*n, delta))
    S = myfunc(ff, params)

    S_old = S.copy()

    i = 2
    ff_fold = np.arange((i-1)*n+1, i*n) / (2*n) / delta
    S_zero = [myfunc(np.floor(i/2)/delta, params)]
    S_nyq = [myfunc((np.floor((i-1)/2) + 0.5)/delta, params)]
    S_nff = myfunc(ff_fold, params)
    S_ff = S_nff[::-1]
    S += np.concatenate([S_zero, S_ff, S_nyq, S_nff])

    while np.mean(S - S_old) > tol: 
        i += 1

        S_old = S.copy()

        ff_fold = np.arange((i-1)*n+1, i*n) / (2*n) / delta

        S_zero = [myfunc(np.floor(i/2)/delta, params)]
        S_nyq = [myfunc((np.floor((i-1)/2) + 0.5)/delta, params)]

        if ut.is_even(i):
            S_nff = myfunc(ff_fold, params)
            S_ff = S_nff[::-1]
        else:
            S_ff = myfunc(ff_fold, params)
            S_nff = S_ff[::-1]

        S += np.concatenate([S_zero, S_ff, S_nyq, S_nff])

    print("Converged with " + str(i) + " iterations")

    return S