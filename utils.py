import numpy as np
import scipy.integrate as integrate


def func_B(x, a, b):
    integrand = lambda y: np.power(y, a-1)*np.power(1-y, b-1)
    return integrate.quad(integrand, 0, x)