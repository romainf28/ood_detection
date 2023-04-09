from sklearn.base import ClassifierMixin
from typing import Optional
import numpy as np


from utils.my_utils import softmax, energy_score, mahalanobis_distance
from ai_irw import AI_IRW


class OODDetector(ClassifierMixin):
    '''
    Args
    - similarity measure : measures the similarity between a test sample and a training dataset.
      Available options : IRW, MSP, E, mahalanobis
    - T : temperature parameter used in normalization formula if needed
    '''

    def __init__(self, base_distrib: Optional[np.ndarray], similarity_measure: str = 'IRW', T: float = 1.0, gamma: float = 1.0):
        super().__init__()

        assert similarity_measure in (
            'MSP', 'IRW', 'E', 'mahalanobis'), 'Please choose one of the available similarity measures.'

        if base_distrib is None:
            assert similarity_measure in (
                'MSP', 'E'), 'To use {} similarity, please provide a train distribution.'.format(similarity_measure)

        self.similarity_measure = similarity_measure
        self.base_distrib = base_distrib
        self.similarity_function_fitted = False
        self.gamma = gamma
        self.T = T

    def fit_similarity_function(self) -> None:
        '''
        Assigns a function which computes the similarity between a data point and the base distribution to the 'similarity_score' property.
        '''
        if self.similarity_measure == 'MSP':
            self.compute_similarity_score = lambda x: 1 - np.max(
                softmax(x), axis=-1)

        elif self.similarity_measure == 'E':
            def e_score_wrapper(x):
                return energy_score(x, temperature=self.T)
            # normalize energy score to make it positive
            self.compute_similarity_score = lambda x: np.max(
                e_score_wrapper(x)) - e_score_wrapper(x)

        elif self.similarity_measure == 'mahalanobis':
            if len(self.base_distrib.shape) == 3:
                self.base_distrib = np.mean(self.base_distrib, axis=-1)
            self.compute_similarity_score = lambda x: mahalanobis_distance(
                x, self.base_distrib)

        elif self.similarity_measure == 'IRW':
            if len(self.base_distrib.shape) == 3:
                self.base_distrib = np.mean(self.base_distrib, axis=-1)
            self.compute_similarity_score = lambda x: 1 - AI_IRW(
                X=self.base_distrib, X_test=x, n_dirs=int(1e3))

    def fit(self):
        self.fit_similarity_function()
        self.similarity_function_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.similarity_function_fitted, "Please fit the similarity function before trying to make predictions"
        if len(X.shape) == 1:
            X = X[None, :]
        return (self.compute_similarity_score(X) <= self.gamma).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.similarity_function_fitted, "Please fit the similarity function before trying to make predictions"
        if len(X.shape) == 1:
            X = X[None, :]
        return self.compute_similarity_score(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return np.mean((preds == y).astype(int))
