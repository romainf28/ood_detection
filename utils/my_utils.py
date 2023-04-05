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
        dim), cov=np.eye(dim), size=n_samples)
    return U/U.sum(axis=1, keepdims=True)


def IRW(X_train: np.ndarray, X_test: np.ndarray, n_proj: Optional[int], random_seed: Optional[int]) -> np.ndarray:
    '''
    Computes the IRW score of the elements of X_test with respect to X_train
    Arguments :
    - X_train : training distribution
    - X_test : test set containing the samples for which we want to compute the IRW score
    - n_proj : The number of directions that we want to use to approximate the unit sphere. If not explicitely set,
    n_proj is defined as 100 times the number of features.
    - random_seed : the random seed utilized to sample the unit sphere
    '''
    if random_seed is None:
        random_seed = 0

    np.random.seed(random_seed)

    X_reduced = X_train.copy()
    X_test_reduced = X_test.copy()

    # if not set, we choose a number of projections equal to 100*nb_features, as suggested in the following paper :
    # https://arxiv.org/pdf/2106.11068.pdf
    if n_proj is None:
        n_proj = X_reduced.shape[1]*100

    U = sample_unit_sphere(n_samples=n_proj, dim=X_reduced.shape[1])

    depth = np.zeros(X_test_reduced.shape)

    projections = np.dot(X_reduced, U.T)
    projections_test = np.dot(X_test_reduced, U.T)

    projections.sort(axis=0)
    for j in range(depth.shape[1]):
        for i in range(depth.shape[0]):
            elt_to_insert = projections_test[i, j]
            insertion_index = 0
            while insertion_index < depth.shape[0]:
                if elt_to_insert > projections_test[insertion_index]:
                    insertion_index += 1
            depth[i, j] = insertion_index

    depth = depth/float(X_reduced.shape[0])

    irw = np.mean(np.minimum(depth, 1-depth), axis=1)

    return irw
