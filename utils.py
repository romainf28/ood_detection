import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x)
    return np.diag(1/np.sum(exp_x, axis=-1)).dot(exp_x)


def logSumExp(X: np.ndarray) -> np.ndarray:
    '''
    X is a 2D-array. Returns the log of the sum of the exponential of X lines
    '''
    return np.log(np.sum(np.exp(X), axis=1))


def mahalanobis_distance(x: np.ndarray, D: np.ndarray) -> np.ndarray:
    '''
    D is a 2D array representing the train distribution. x is a vector representing the test sample.
    Computes the Mahalanobis distance between x and D.
    '''
    m = np.mean(D, axis=0).reshape(1, D.shape[1])
    VI = np.linalg.inv(np.cov(D.T))
    return cdist(XA=np.mean(x, axis=-1), XB=m, metric='mahalanobis', VI=VI)


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
