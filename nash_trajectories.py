# g solver

from scipy.integrate import solve_ivp
from coupled_ricatti import T, kappa, fun

n = 1
g=10

sol_ricatti = solve_ivp(fun, (T, 0), [kappa, 0], dense_output=True, rtol=1e-10, atol=1e-12)

def traj(t, g):
    alpha, beta = sol_ricatti.sol(t)
    dg = - (g/kappa) * (alpha - 2*n*beta / (n + 1))
    return dg


sol_g = solve_ivp(traj, t_span=(0, T), y0=[g], dense_output=True, rtol=1e-10, atol=1e-12)

print(sol_g.y)

