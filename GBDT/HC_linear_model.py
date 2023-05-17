import numpy as np


def hc_linear(p, v, f):
    # p represents position, v for velocity and f for force feedback
    reg = np.mat(np.zeros(1, 3))
    y = []
    for i in range(0, len(p)):
        reg = np.row_stack((reg, [1, v[i], np.log(p[i])]))
        y.append(np.log(f[i]))
    return np.delete(reg, 0, axis=0), y

