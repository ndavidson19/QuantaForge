# cy_strategy.pyx
import numpy as np
cimport numpy as np

def generate_signals(np.ndarray[np.float64_t, ndim=1] close_prices):
    cdef int n = close_prices.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] signals = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close_prices[i] > close_prices[i - 1]:
            signals[i] = 1.0
        else:
            signals[i] = -1.0
    return signals
