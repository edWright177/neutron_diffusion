import os
import numpy as np
from solver2d import solve_fixed_source, plot_field

os.makedirs("figures", exist_ok=True)

Lx = 10.0
Ly = 6.0
Nx = 140
Ny = 84


def gaussian_source(X, Y, x0, y0, strength=8.0, sigma=0.5):
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    return strength * np.exp(-r2 / (2.0 * sigma**2))


def D_material(X, Y):
    D = 1.2 * np.ones_like(X)
    moderator = (X > 4.0) & (X < 8.5) & (Y > 1.0) & (Y < 5.0)
    D[moderator] = 1.8
    return D


def sigma_material(X, Y):
    sigma = 0.12 * np.ones_like(X)
    absorber = (X > 6.2) & (X < 6.8) & (Y > 1.2) & (Y < 4.8)
    sigma[absorber] = 1.1
    return sigma


def source_case(X, Y):
    return gaussian_source(X, Y, x0=2.4, y0=3.0, strength=8.5, sigma=0.45)


X, Y, phi = solve_fixed_source(
    Lx=Lx,
    Ly=Ly,
    Nx=Nx,
    Ny=Ny,
    D_fn=D_material,
    sigma_a_fn=sigma_material,
    source_fn=source_case,
)

plot_field(
    X, Y, phi,
    title="Fixed-source neutron diffusion with heterogeneous materials",
    filename="figures/fixed_source_heterogeneous.png",
)