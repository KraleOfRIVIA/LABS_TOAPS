import numpy as np
import matplotlib.pyplot as plt
def modified_Euler(function, y_matrix, h):
    y = np.zeros((np.size(h), np.size(y_matrix)))  # creates the matrix that we will fill
    y[0, :] = y_matrix # sets the initial values of the matrix

    for i in range(len(time) - 1):  # apply the Euler
        dt = time[i + 1] - time[i]  # Step size
        k1 = f(y[i], t[i]) * dt
        k2 = f(y[i] + k1, t[i + 1]) * dt
        y[i+1] = y[i] + 0.5*dt * (3*f[i] - f[i-1])

    return y