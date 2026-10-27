import numpy as np
from random import sample


def init_centroids(k: int, inputs: np.ndarray) -> np.ndarray:
    """
    Selects k random rows from inputs and returns them as the chosen centroids.
    Hint: use random.sample

    Parameters
    ----------
    k : int
        Number of cluster centroids.
    inputs : np.ndarray
        A 2D Numpy array, each row of which is one input.

    Returns
    -------
    np.ndarray
        A Numpy array of k cluster centroids, one per row.
    """

    chosen_indices = sample(range(inputs.shape[0]), k)
    return inputs[chosen_indices]


def assign_step(inputs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance.

    Parameters
    ----------
    inputs : np.ndarray
        Inputs of data, a 2D Numpy array.
    centroids : np.ndarray
        A Numpy array of k current centroids.

    Returns
    -------
    np.ndarray
        A Numpy array of centroid indices, one for each row of the inputs.
    """
    squared_distances = np.sum(
        (inputs[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(squared_distances, axis=1)


def update_step(inputs: np.ndarray, indices: np.ndarray, k: int) -> np.ndarray:
    """
        Computes the centroid for each cluster.
    2
        Parameters
        ----------
        inputs : np.ndarray
            Inputs of data, a 2D Numpy array.
        indices : np.ndarray
            A Numpy array of centroid indices, one for each row of the inputs.
        k : int
            Number of cluster centroids.

        Returns
        -------
        np.ndarray
            A Numpy array of k cluster centroids, one per row.
    """
    centroids = np.zeros((k, inputs.shape[1]))

    for cluster_idx in range(k):
        cluster_points = inputs[indices == cluster_idx]
        if cluster_points.shape[0] > 0:
            centroids[cluster_idx] = np.mean(cluster_points, axis=0)
        else:
            random_idx = np.random.randint(0, inputs.shape[0])
            centroids[cluster_idx] = inputs[random_idx]

    return centroids


def kmeans(inputs: np.ndarray, k: int, max_iter: int, tol: float) -> np.ndarray:
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement.

    Parameters
    ----------
    inputs : np.ndarray
        Inputs of data, a 2D Numpy array.
    k : int
        Number of cluster centroids.
    max_iter : int
        The maximum number of times the algorithm can iterate trying to optimize the centroid values.
    tol : float
        The tolerance we determine convergence with when compared to the ratio as stated on handout.

    Returns
    -------
    np.ndarray
        A Numpy array of k cluster centroids, one per row.
    """
    centroids = init_centroids(k, inputs)

    for _ in range(max_iter):
        indices = assign_step(inputs, centroids)
        new_centroids = update_step(inputs, indices, k)

        denominator = np.linalg.norm(centroids)
        numerator = np.linalg.norm(new_centroids - centroids)
        ratio = numerator / denominator if denominator != 0 else numerator

        centroids = new_centroids
        if ratio < tol:
            break

    return centroids
