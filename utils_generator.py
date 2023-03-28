import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional
import torch


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


def renyi_divergence(p: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    '''
    p and q are array representing the distribution. x is the parameter of renyi divergence.
    Computes the Renyi divergence between p and q.
    '''
    return (1/(1-alpha)) * np.log(np.sum((p**alpha)/(q**(alpha-1))))


def FR_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    '''
    p and q are array representing the distribution. x is the parameter of renyi divergence.
    Computes the Fischer Rao distance between p and q.
    '''
    return (2/np.pi) * np.arccos(np.sum((p*q)**0.5))


def set_norm_logits(x: torch.tensor, model:torch.model, max_token: int, voc:torch.tensor) -> np.ndarray:
    set_norm_logits = []
    while len(set_norm_logits)<max_token :
        if len(set_norm_logits) >0:
            y = voc[set_norm_logits[-1].multinomial(num_samples=1, replacement=True)]
            set_norm_logits.append(torch.nn.softmax(model(torch.cat((x,y),0))))
        else:
            set_norm_logits.append(torch.nn.softmax(model(x)))
        
    return np.ndarray(set_norm_logits)

def anomaly_score(x: torch.tensor, set_norm_logits:np.ndarray, divergence:str, alpha:Optional[float])->int:
    cum_div = 0
    u = np.ones(len(set_norm_logits[0]))/len(set_norm_logits[0])
    for p in set_norm_logits:
        if p == "renyi":
            cum_div += renyi_divergence(p,u,alpha)
        else:
            cum_div += FR_distance(p,u)
    return (1/len(set_norm_logits)) * cum_div
