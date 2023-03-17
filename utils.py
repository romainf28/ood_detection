import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional


def softmax(x: np.ndarray) -> np.ndarray:
    '''
    Returns the softmax of X.
    '''
    exp_x = np.exp(x)
    return np.diag(1/np.sum(exp_x, axis=-1)).dot(exp_x)


def logSumExp(X: np.ndarray) -> np.ndarray:
    '''
    Returns the log of the sum of the exponential of X lines
    '''
    return np.log(np.sum(np.exp(X), axis=1))


def energy_score(X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    '''
    Returns the energy based score with temperature T
    '''
    return temperature*logSumExp(X/temperature)


def mahalanobis_distance(x: np.ndarray, D: np.ndarray) -> np.ndarray:
    '''
    D is a 2D array representing the train distribution. x is a vector representing the test sample.
    Computes the Mahalanobis distance between x and D.
    '''
    m = np.mean(D, axis=0).reshape(1, D.shape[1])
    VI = np.linalg.inv(np.cov(D.T))
    return cdist(XA=np.mean(x, axis=-1), XB=m, metric='mahalanobis', VI=VI)


def sample_unit_sphere(n_samples: int, dim: int) -> np.ndarray:
    '''
    Returns n_samples uniformly distributed points on the unit sphere in dimension d
    '''
    U = np.random.multivariate_normal(mean=np.zeros(
        dim), cov=np.eye((dim, dim)), size=n_samples)
    return U/U.sum(axis=1, keepdims=True)


def IRW(X_train: np.ndarray, X_test: Optional[np.ndarray], n_proj: Optional[int], random_seed: Optional[int]) -> np.ndarray:
    if random_seed is None:
        random_seed = 0

    np.random.seed(random_seed)

    if X_test is None:
        X_reduced = X_train.copy()

    # if not set, we choose a number of projections equal to 100*nb_features, as suggested in the following paper :
    # https://arxiv.org/pdf/2106.11068.pdf
    if n_proj is None:
        n_proj = X_reduced.shape[1]*100

    U = sample_unit_sphere(n_samples=n_proj, dim=X_reduced.shape[1])
