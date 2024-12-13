import numpy as np
import scipy

# time convolution with trapezoid rule, assuming output starts at 0
def conv_trap(K, x, t):
    Nt = len(t)
    dt = t[1] - t[0]
    y = scipy.signal.convolve(np.pad(K, (Nt-1, 0)), x, mode='valid')*dt
    y -= (K[0]*x/2 + x[0]*K/2)*dt
    return y

# Solve y = c0x - K*x for x
def solve_volterra_second_type(K, c0, y, t):
    Nt = len(t)
    dt = t[1] - t[0]
    
    Kmat = scipy.linalg.toeplitz(K, np.zeros(Nt))
    Kmat[:, 0] /= 2
    np.fill_diagonal(Kmat, np.diag(Kmat)/2)

    if c0 == 0:
        assert(y[0] == 0)
        Kmat[0, 0] = K[0]
    else:
        Kmat[0, 0] = 0

    x = scipy.linalg.solve_triangular(c0*np.eye(Nt) - Kmat*dt, y, lower=True, check_finite=False)
    return x