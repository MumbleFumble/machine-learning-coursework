"""
This file contains some rudimentary test cases with which to check your
implementations in eigenvalues.py

Brown cs1420, Spring 2026
"""

import numpy as np
from eigenvalues import qr, compute_eigenvalues


def generate_real_unique_eigenvalue_matrix(size=25, seed=1420):
    """
    Generates a square matrix with eigenvalues 1 through size, inclusive.

    Parameters
    ----------
    size : int, optional
        Size of the square matrix, by default 25
    seed : int, optional
        Random seed, by default 42

    Returns
    -------
    np.ndarray
        Square matrix with real, unique eigenvalues
    """
    np.random.seed(seed)
    eigenvalues = np.linspace(1, size, size)
    random_matrix = np.random.rand(size, size) * 100
    q, _ = np.linalg.qr(random_matrix)
    matrix = q @ np.diag(eigenvalues[:size]) @ np.linalg.inv(q)
    return matrix


def test_qr_decomposition(n: int) -> bool:
    """Routine to test the QR decomposition implementation
    against Numpy's implementation

    Parameters
    ----------
    n : int
        Matrix size parameter

    Returns
    -------
    bool
        If all cases pass, returns true. Otherwise, exits.
    """
    np.random.seed(1420)
    B = np.random.randn(n, n)
    B = B.T @ B
    Q_test, R_test = qr(B)
    Q_actual, R_actual = np.linalg.qr(B)

    if not (np.allclose(Q_test @ R_test, B, rtol=1e-6)):
        print("\nFailed QR decomposition recovery test on symmetric random matrix!")
        exit()

    if not (
        np.allclose(np.abs(Q_test), np.abs(Q_actual), rtol=1e-6)
        and np.allclose(np.abs(R_test), np.abs(R_actual), rtol=1e-6)
    ):
        print("\nFailed QR decomposition match test on symmetric random matrix!")
        exit()

    return True


def test_eigenvalues(n: int) -> bool:
    """Routine to test the eigenvalue calculation implementation
    against Numpy's implementation

    Parameters
    ----------
    n : int
        Matrix size parameter

    Returns
    -------
    bool
        If all cases pass, returns true. Otherwise, false.
    """
    B = generate_real_unique_eigenvalue_matrix(n, seed=1420)
    eigs_test = np.sort(compute_eigenvalues(B, 1000))
    eigs_actual = np.sort(np.linalg.eigvals(B))
    if not (np.allclose(eigs_test, eigs_actual, rtol=1e-6)):
        print("\nFailed eigenvalue match test on random matrix!")
        exit(1)
    return True


if __name__ == "__main__":
    print("\rTesting QR decomposition implementation...")
    for i, size in enumerate([4, 12, 15, 100, 500]):
        print(f"\rqr test {i+1}/6 on size {size}x{size}", end="")
        result = test_qr_decomposition(size)

    print("\rTesting eigenvalue computation implementation...")
    for i, size in enumerate([4, 12, 15, 25, 50]):
        print(f"\rcompute_eigenvalues test {i+1}/5 on size {size}x{size}", end="")
        result = test_eigenvalues(size)

    print("\rPassed all cases!" + " " * 40)
