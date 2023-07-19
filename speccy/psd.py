
import numpy as np
from scipy.special import gamma

def jonswap(f, params):
    
    alpha = params[0]
    Tp = params[1]
    gamma = params[2]
    r = params[3]
    
    sig1 = 0.07
    sig2 = 0.09
    s = 4

    sigma = f.copy()
    sigma[f <= 1/Tp] = sig1
    sigma[f > 1/Tp] = sig2

    delta = np.exp(-1/(2*sigma**2) * (f*Tp - 1) ** 2)

    S = alpha * (2*np.pi*f) ** (-r) * np.exp(-r/s * (f*Tp) ** (-s)) * gamma ** delta
    return(S)

def matern(ff, params):
    """General Matern PSD a la Lilly/Sykulski"""
    
    eta = params[0]
    alpha = params[1]
    lmbda = params[2]

    c = (gamma(1/2) * gamma(alpha - 1/2)) / (2 * gamma(alpha) * np.pi)
    S = np.power(eta, 2) * np.power(lmbda, 2*alpha - 1) / c
    S *= np.power(np.power(2*np.pi*ff, 2) + np.power(lmbda, 2), -alpha)
    
    return S