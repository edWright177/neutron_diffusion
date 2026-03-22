import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve, eigsh


def make_grid_2d(Lx: float, Ly: float, Nx: int, Ny: int):
    x = np.linspace(0.0, Lx, Nx + 1)
    y = np.linspace(0.0, Ly, Ny + 1)
    hx = Lx / Nx
    hy = Ly / Ny
    X, Y = np.meshgrid(x, y, indexing="xy")
    return x, y, X, Y, hx, hy


def interior_mesh(Lx: float, Ly: float, Nx: int, Ny: int):
    x, y, _, _, hx, hy = make_grid_2d(Lx, Ly, Nx, Ny)
    xi = x[1:-1]
    yi = y[1:-1]
    Xi, Yi = np.meshgrid(xi, yi, indexing="xy")
    return xi, yi, Xi, Yi, hx, hy


def flatten_index(i: int, j: int, nx_int: int) -> int:
    return j * nx_int + i


def harmonic_mean(a, b, eps=1e-14):
    return 2.0 * a * b / (a + b + eps)


def assemble_diffusion_matrix(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    D_fn,
    sigma_a_fn,
):
    """
    Assemble A for
        -div(D grad phi) + Sigma_a phi
    with Dirichlet boundary conditions on a rectangular grid.
    """
    xi, yi, Xi, Yi, hx, hy = interior_mesh(Lx, Ly, Nx, Ny)

    D = D_fn(Xi, Yi)
    sigma_a = sigma_a_fn(Xi, Yi)

    nx_int = Nx - 1
    ny_int = Ny - 1
    n = nx_int * ny_int

    A = lil_matrix((n, n), dtype=float)

    for j in range(ny_int):
        for i in range(nx_int):
            k = flatten_index(i, j, nx_int)

            Dij = D[j, i]

            # Face diffusion coefficients via harmonic averaging
            Dw = harmonic_mean(Dij, D[j, i - 1]) if i > 0 else Dij
            De = harmonic_mean(Dij, D[j, i + 1]) if i < nx_int - 1 else Dij
            Ds = harmonic_mean(Dij, D[j - 1, i]) if j > 0 else Dij
            Dn = harmonic_mean(Dij, D[j + 1, i]) if j < ny_int - 1 else Dij

            cx_w = Dw / hx**2
            cx_e = De / hx**2
            cy_s = Ds / hy**2
            cy_n = Dn / hy**2

            A[k, k] = cx_w + cx_e + cy_s + cy_n + sigma_a[j, i]

            if i > 0:
                A[k, flatten_index(i - 1, j, nx_int)] = -cx_w
            if i < nx_int - 1:
                A[k, flatten_index(i + 1, j, nx_int)] = -cx_e
            if j > 0:
                A[k, flatten_index(i, j - 1, nx_int)] = -cy_s
            if j < ny_int - 1:
                A[k, flatten_index(i, j + 1, nx_int)] = -cy_n

    return Xi, Yi, csr_matrix(A)


def assemble_fixed_source_system(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    D_fn,
    sigma_a_fn,
    source_fn,
):
    xi, yi, Xi, Yi, _, _ = interior_mesh(Lx, Ly, Nx, Ny)
    _, _, A = assemble_diffusion_matrix(Lx, Ly, Nx, Ny, D_fn, sigma_a_fn)
    source = source_fn(Xi, Yi)
    b = source.reshape(-1).copy()
    return Xi, Yi, A, b


def solve_fixed_source(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    D_fn,
    sigma_a_fn,
    source_fn,
):
    x, y, X, Y, _, _ = make_grid_2d(Lx, Ly, Nx, Ny)
    _, _, A, b = assemble_fixed_source_system(
        Lx, Ly, Nx, Ny, D_fn, sigma_a_fn, source_fn
    )

    phi_int = spsolve(A, b)

    phi = np.zeros((Ny + 1, Nx + 1), dtype=float)
    phi[1:-1, 1:-1] = phi_int.reshape((Ny - 1, Nx - 1))
    return X, Y, phi


def assemble_fission_matrix(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    nu_sigma_f_fn,
):
    xi, yi, Xi, Yi, _, _ = interior_mesh(Lx, Ly, Nx, Ny)
    nu_sigma_f = nu_sigma_f_fn(Xi, Yi).reshape(-1)
    F = diags(nu_sigma_f, offsets=0, format="csr")
    return Xi, Yi, F


def solve_criticality(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    D_fn,
    sigma_a_fn,
    nu_sigma_f_fn,
    tol: float = 1e-8,
    max_iters: int = 200,
):
    """
    Solve the one-group criticality problem

        -div(D grad phi) + Sigma_a phi = (1/k) nuSigma_f phi

    using power iteration on A^{-1} F.

    Returns:
        X, Y, k_eff, phi
    """
    x, y, X, Y, _, _ = make_grid_2d(Lx, Ly, Nx, Ny)

    xi, yi, A = assemble_diffusion_matrix(Lx, Ly, Nx, Ny, D_fn, sigma_a_fn)
    _, _, F = assemble_fission_matrix(Lx, Ly, Nx, Ny, nu_sigma_f_fn)

    n = A.shape[0]

    # Initial guess: positive flux everywhere in the interior
    phi = np.ones(n, dtype=float)
    phi /= np.linalg.norm(phi)

    k_eff = 1.0

    for it in range(max_iters):
        phi_old = phi.copy()
        k_old = k_eff

        # F phi_old
        rhs = F @ phi_old

        # Avoid total-zero RHS if fission region is empty
        rhs_norm = np.linalg.norm(rhs)
        if rhs_norm == 0.0:
            raise ValueError("Fission operator produced zero vector. Check nu_sigma_f_fn.")

        # Solve A y = F phi_old
        y = spsolve(A, rhs)

        # Rayleigh-style update for k
        num = np.dot(phi_old, F @ phi_old)
        den = np.dot(phi_old, F @ y)

        if abs(den) < 1e-14:
            raise ValueError("Encountered near-zero denominator in k update.")

        k_eff = num / den

        # Normalize flux shape
        phi = y / np.linalg.norm(y)

        # Keep sign consistent
        if phi.sum() < 0:
            phi *= -1.0

        # Convergence check
        flux_err = np.linalg.norm(phi - phi_old)
        k_err = abs(k_eff - k_old)

        if flux_err < tol and k_err < tol:
            print(f"Power iteration converged in {it + 1} iterations.")
            break
    else:
        print("Warning: power iteration reached max_iters without full convergence.")

    phi_grid = np.zeros((Ny + 1, Nx + 1), dtype=float)
    phi_grid[1:-1, 1:-1] = phi.reshape((Ny - 1, Nx - 1))

    # Normalize for plotting
    phi_grid /= np.max(np.abs(phi_grid))

    return X, Y, k_eff, phi_grid

def plot_field(X, Y, field, title, cmap="viridis", filename=None):
    plt.figure(figsize=(8, 5.5))
    pcm = plt.pcolormesh(X, Y, field, shading="auto", cmap=cmap)
    plt.colorbar(pcm)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200)
    plt.show()


def plot_contours(X, Y, field, title, levels=20, cmap="viridis", filename=None):
    plt.figure(figsize=(8, 5.5))
    cs = plt.contourf(X, Y, field, levels=levels, cmap=cmap)
    plt.colorbar(cs)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200)
    plt.show()