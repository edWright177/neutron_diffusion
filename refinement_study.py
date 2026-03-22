import numpy as np
import matplotlib.pyplot as plt
from solver2d import solve_fixed_source

Lx = 10.0
Ly = 6.0


def gaussian_source(X, Y, x0, y0, strength=8.0, sigma=0.5):
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    return strength * np.exp(-r2 / (2.0 * sigma**2))


def D_material(X, Y):
    D = 1.2 * np.ones_like(X)
    D[(X > 4.0) & (X < 8.5) & (Y > 1.0) & (Y < 5.0)] = 1.8
    return D


def sigma_material(X, Y):
    sigma = 0.12 * np.ones_like(X)
    sigma[(X > 6.2) & (X < 6.8) & (Y > 1.2) & (Y < 4.8)] = 1.1
    return sigma


def source_case(X, Y):
    return gaussian_source(X, Y, x0=2.4, y0=3.0, strength=8.5, sigma=0.45)


resolutions = [(40, 24), (80, 48), (120, 72), (160, 96)]
peak_flux = []
cell_sizes = []

for Nx, Ny in resolutions:
    X, Y, phi = solve_fixed_source(
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        D_fn=D_material,
        sigma_a_fn=sigma_material,
        source_fn=source_case,
    )
    peak_flux.append(phi.max())
    cell_sizes.append(Lx / Nx)
    print(f"Nx={Nx:3d}, Ny={Ny:3d}, h={Lx/Nx:.5f}, max(phi)={phi.max():.6f}")

plt.figure(figsize=(7, 4.5))
plt.plot(cell_sizes, peak_flux, "o-", linewidth=2)
plt.gca().invert_xaxis()
plt.xlabel("Grid spacing h")
plt.ylabel("Peak flux")
plt.title("Grid refinement study")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/grid_refinement_peak_flux.png", dpi=200)
plt.show()