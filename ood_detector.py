from sklearn.base import ClassifierMixin
from typing import Optional
import numpy as np


from utils import softmax, energy_score, mahalanobis_distance


class OODDetector(ClassifierMixin):
    '''
    Args
    - similarity measure : measures the similarity between a test sample and a training dataset.
      Available options : IRW, MSP, E, mahalanobis
    - T : temperature parameter used in normalization formula if needed
    '''

    def __init__(self, base_distrib: Optional[np.ndarray], base_ood_distrib: Optional[np.ndarray], similarity_measure: str = 'IRW', T: float = 1.0):
        super().__init__()

        assert similarity_measure in (
            'MSP', 'IRW', 'E', 'mahalanobis'), 'Please choose one of the available similarity measures.'

        if base_distrib is None:
            assert similarity_measure in (
                'MSP', 'E'), 'To use {} similarity, please provide a train distribution.'.format(similarity_measure)

        self.similarity_measure = similarity_measure
        self.base_distrib = base_distrib
        self.base_ood_distrib = base_ood_distrib

    def similarity_function(self) -> None:
        '''
        Assigns a function which computes the similarity between a data point and the base distribution to the 'similarity_score' property.
        '''
        if self.similarity_measure == 'MSP':
            self.similarity_score = lambda x: 1-np.max(softmax(x), axis=-1)

        elif self.similarity_measure == 'E':
            self.similarity_score = lambda x: energy_score(
                x, temperature=self.T)

        elif self.similarity_measure == 'mahalanobis':
            self.similarity_score = lambda x: mahalanobis_distance(
                x, self.base_distrib)
