import os
import numpy as np
from solver2d import solve_criticality, plot_field

os.makedirs("figures", exist_ok=True)

Lx = 10.0
Ly = 6.0
Nx = 140
Ny = 84


def D_core(X, Y):
    D = 1.3 * np.ones_like(X)
    reflector = (X > 8.5)
    D[reflector] = 2.0
    return D


def sigma_a_core(X, Y):
    sigma = 0.10 * np.ones_like(X)

    # stronger absorption strip (control-rod-like)
    rod = (X > 5.8) & (X < 6.3) & (Y > 1.2) & (Y < 4.8)
    sigma[rod] = 0.65

    # reflector region
    reflector = (X > 8.5)
    sigma[reflector] = 0.04
    return sigma


def nu_sigma_f_core(X, Y):
    nu_sigma_f = np.zeros_like(X)

    # fissile core region
    core = (X > 1.0) & (X < 8.3) & (Y > 0.8) & (Y < 5.2)
    nu_sigma_f[core] = 0.18

    # weakened fission in rod region
    rod = (X > 5.8) & (X < 6.3) & (Y > 1.2) & (Y < 4.8)
    nu_sigma_f[rod] = 0.05
    return nu_sigma_f


X, Y, k_eff, phi = solve_criticality(
    Lx=Lx,
    Ly=Ly,
    Nx=Nx,
    Ny=Ny,
    D_fn=D_core,
    sigma_a_fn=sigma_a_core,
    nu_sigma_f_fn=nu_sigma_f_core,
)

print(f"k_eff = {k_eff:.6f}")

plot_field(
    X, Y, phi,
    title=f"Dominant flux mode, k_eff = {k_eff:.5f}",
    filename="figures/criticality_mode.png",
)