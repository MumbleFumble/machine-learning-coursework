"""
This file contains an implementation of the QR decomposition and
a routine to approximate the eigenvalues of a given matrix.

Brown cs1420, Spring 2026
"""

import numpy as np


def sign(x: float) -> int:
    """
    Returns the sign of the input, with the convention that sign(0) = +1.
    """
    if x >= 0:
        return 1
    else:
        return -1


def qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the QR decomposition of A using the Householder algorithm.
    Corresponds to the pseudocode for Algorithm 1, Problem 5.

    Parameters
    ----------
    A : np.ndarray
        N x N square matrix.

    Returns
    -------
    np.ndarray
        Q, an orthogonal matrix such that A = QR
    np.ndarray
        R, an upper triangular matrix such that A = QR
    """
    n = A.shape[0]

    # TODO: Define the matrices needed for the Householder algorithm (lines 2-4 of Algorithm 1).
    R = None
    V = None
    I = None
    pass

    for k in range(n):
        # TODO: Implement lines 6-9 of the Householder algorithm (Algorithm 1).
        pass

    Q = np.zeros((n, n))
    for j in range(n):
        x = I[:, j]
        for k in reversed(range(n)):
            x[k:] = x[k:] - 2 * V[k:, k].reshape(-1, 1) @ (
                V[k:, k].reshape(-1, 1).T @ x[k:]
            )
        Q[:, j] = x
    return Q, R


def compute_eigenvalues(A: np.ndarray, m: int = 500) -> np.ndarray:
    """
    Computes the eigenvalues of A using the QR algorithm.
    Corresponds to the pseudocode for Algorithm 2, Problem 5.

    Parameters
    ----------
    A : np.ndarray
        N x N square matrix.
    m : int
        Total number of iterations.

    Returns
    -------
    np.ndarray
        1-dimensional ndarray representing the n eigenvalues of A.
    """
    # TODO: Implement Algorithm 2.
    pass


def generate_real_unique_eigenvalue_matrix(
    size: int = 25, seed: int = 42
) -> np.ndarray:
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
    random_matrix = np.random.rand(size, size)
    q, _ = np.linalg.qr(random_matrix)
    matrix = q @ np.diag(eigenvalues[:size]) @ np.linalg.inv(q)
    return matrix
