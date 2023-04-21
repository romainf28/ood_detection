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
    return cdist(x, XB=m, metric='mahalanobis', VI=VI)

