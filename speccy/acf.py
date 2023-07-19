
import numpy as np
from scipy.special import kv as K_nu
from scipy.special import gamma

def matern(dx, params, sigma = 0, acf = True):
    """General Matern covariance a la Lilly/Sykulski"""

    eta = params[0]
    alpha = params[1]
    lmbda = params[2]

    nu = alpha - 1/2

    K = 2 * np.power(eta, 2) / (gamma(nu) * np.power(2, nu))
    K *= np.power(np.abs(lmbda * dx), nu)
    K *= K_nu(nu, np.abs(lmbda * dx))
    K[np.isnan(K)] = np.power(eta, 2.)

    if acf:
        K[0] = K[0] + sigma**2
    else:
        n = dx.shape[0]
        K += sigma**2 * np.eye(n)
    
    return K