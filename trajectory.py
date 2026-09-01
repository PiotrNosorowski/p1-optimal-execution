from ricatti import solver, kappa, T, g0
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

# interpolation function requires an increase of values
# so arrays must be inverted 
a = solver.y[0]                   # as a is the cost of the remaining game
a = a[::-1]

t = solver.t              # as t is the time remaining
t = t[::-1]

# an interpolation function; (x,y) as input
cs = CubicSpline(t, a)

# to calculate g's rate of change
def dynamics(t, g):
    # u* = (a/kappa) * g
    # dg/dt = -u*
    return -(cs(t)/kappa) * g


ls = np.linspace(0, T, 200)

g_solver = solve_ivp(dynamics, (0, T), y0=[g0], t_eval=ls)

if __name__ == "__main__":
    plt.plot(g_solver.t, g_solver.y[0])
    plt.savefig("g_solver.jpg")
    plt.show()

# print(g_solver.y[0][-1])

# the optimal selling rate
u = (cs(g_solver.t) / kappa) * g_solver.y[0]

if __name__ == "__main__":
    plt.figure()
    plt.plot(g_solver.t, u)
    plt.savefig("u_.jpg")
    plt.show()