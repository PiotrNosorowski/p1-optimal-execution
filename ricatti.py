from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

kappa = 1
phi = 0.25
T = 1
g0 = 10

# rule determining the rate of change of a
def ricatti(t, a):
     return a**2 / kappa - phi

# returns evenly separated samples
# start, stop, n_samples
ls = np.linspace(T, 0, 200)

# to solve the initial value problem
# y0 determines the initial state, which here is a terminal condition
# and a(T) = kappa at starting point
solver = solve_ivp(ricatti, (T, 0), y0=[kappa], t_eval=ls)

if __name__ == "__main__":
    # the curve of cost-to-go
    plt.plot(solver.t, solver.y[0])
    plt.savefig("t1.jpg")
    plt.show()

