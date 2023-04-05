import numpy as np
from typing import Optional, Literal
import torch
from transformers import models, AutoModelForSeq2SeqLM, tokenization_utils_base





def renyi_divergence(p: torch.tensor, q: torch.tensor, alpha: float) -> torch.tensor:
    '''
    p and q are tensors representing the distribution. alpha is the parameter of renyi divergence.
    Computes the Renyi divergence between p and q.
    '''
    return (1/(1-alpha)) * torch.log(torch.sum((p**alpha)/(q**(alpha-1)), dim=2))


def FR_distance(p: torch.tensor, q: torch.tensor) -> torch.tensor:
    '''
    p and q are tensors representing the distribution.
    Computes the Fischer Rao distance between p and q.
    '''
    return (2/np.pi) * torch.arccos(torch.sum((p*q)**0.5, dim=2))


class GenerationModel :
    def __init__(self, model_name:str):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.softmax = torch.nn.Softmax(dim=2)

        self.vocab_size = self.model.config.vocab_size

    def generate_set(self, x:tokenization_utils_base.BatchEncoding):
        outputs = self.model.generate(**x, return_dict_in_generate=True,output_scores=True, num_beams=1)
        output_scores = torch.stack(list(outputs.scores), dim=0).permute(1,0,2)
        return self.softmax(output_scores), outputs.sequences
    
    def anomaly_score(self,x:tokenization_utils_base.BatchEncoding, divergence:Literal["Renyi", "FR"], 
                      alpha:Optional[float])->int:
        set_proba, _ = self.generate_set(x)
        
        uniform_proba = torch.ones(set_proba.shape)/self.vocab_size

        if divergence == "Renyi":
            if not alpha:
                raise AttributeError("When you use Renyi divergence, you must define an alpha")
            div = torch.sum(renyi_divergence(set_proba, uniform_proba, alpha), dim=1)
        elif divergence =="FR":
            div = torch.sum(FR_distance(set_proba, uniform_proba), dim=1)
        else: 
            raise AttributeError("The divergence should be 'Renyi' or 'FR'")
        return (1/set_proba.shape[1]) * div
