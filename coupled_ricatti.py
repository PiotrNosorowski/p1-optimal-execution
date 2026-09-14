from scipy.integrate import solve_ivp
import numpy as np
from ricatti import solver

kappa = 1
phi = 0.25
n = 1                # n=1 for sanity check purposes
T = 10


def fun(t, y):
    alpha = y[0]
    beta = y[1]

    dalpha = alpha**2 / kappa - phi
    dbeta = 1 / kappa * (2 * alpha * beta - (2 * n * beta) ** 2 / (n + 1) ** 2)

    return dalpha, dbeta

# returns evenly separated samples
# start, stop, n_samples
ls = np.linspace(T, 0, 200)

# extended by relative and absolute tolerance to tighten solver accuracy
solve_ivp_c = solve_ivp(fun, t_span=(T, 0), y0=[kappa, 0], t_eval=ls,
                        rtol=1e-10, atol=1e-12)


# sanity check 
print(np.max(np.abs(solve_ivp_c.y[0] - solver.y[0])))



