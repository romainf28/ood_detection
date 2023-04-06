import numpy as np
from typing import Optional, Literal
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, tokenization_utils_base, AutoTokenizer
from tqdm import tqdm



def renyi_divergence(p: torch.tensor, q: torch.tensor, alpha: float) -> torch.tensor:
    '''
    p and q are tensors representing the distribution. alpha is the parameter of renyi divergence.
    Computes the Renyi divergence between p and q.
    '''
    return (1/(alpha-1)) * torch.log(torch.round(torch.sum((p**alpha)/(q**(alpha-1)), dim=2), decimals=4))


def FR_distance(p: torch.tensor, q: torch.tensor) -> torch.tensor:
    '''
    p and q are tensors representing the distribution.
    Computes the Fischer Rao distance between p and q.
    '''
    return (2/np.pi) * torch.arccos(torch.round(torch.sum((p*q)**0.5, dim=2), decimals=4))


class GenerationModel :
    def __init__(self, model_name:str):

        if torch.cuda.is_available():
            self.device = 'cuda'
            
        else:
            self.device = 'cpu'
        print(self.device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")


        self.softmax = torch.nn.Softmax(dim=2).to(self.device)

        self.vocab_size = self.model.config.vocab_size
        self.threshold = None

    def generate_set(self, x:tokenization_utils_base.BatchEncoding, temperature:float=1):
        outputs = self.model.generate(**x.to(self.device), return_dict_in_generate=True,output_scores=True, num_beams=1)
        output_scores = torch.stack(list(outputs.scores), dim=0).permute(1,0,2)
        return self.softmax(output_scores/temperature), outputs.sequences
    
    def anomaly_score(self,x:tokenization_utils_base.BatchEncoding, divergence:Literal["Renyi", "FR"], 
                      scenario:Literal["s0", "s1"], alpha:Optional[float], temperature:float=1)->int:
        set_proba, sequences = self.generate_set(x.to(self.device), temperature=temperature)
        
        if scenario=="s0":
            proba = torch.ones(set_proba.shape)/self.vocab_size
        elif scenario=="s1":
            proba = torch.ones(set_proba.shape)/self.vocab_size #TODO

        if divergence == "Renyi":
            if not alpha:
                raise AttributeError("When you use Renyi divergence, you must define an alpha")
            div = torch.sum(renyi_divergence(set_proba, proba, alpha), dim=1)
        elif divergence =="FR":
            div = torch.sum(FR_distance(set_proba, proba), dim=1)
        else: 
            raise AttributeError("The divergence should be 'Renyi' or 'FR'")
        return (1/set_proba.shape[1]) * div, sequences
    
    def define_threshold(self, data_loader:DataLoader, r:float, divergence:Literal["Renyi", "FR"], 
                         scenario:Literal["s0", "s1"], alpha:Optional[float], temperature:float=1):
        list_anomaly = []
        for _, batch in tqdm(enumerate(data_loader), desc="Computing anomaly score for each data", total=data_loader.__len__()):
            x = self.tokenizer(batch["sourceString"], return_tensors="pt", padding=True).to(self.device)
            anomaly, _ = self.anomaly_score(x, divergence, scenario, alpha, temperature)
            list_anomaly.append(anomaly)
        self.threshold = torch.quantile(torch.cat(list_anomaly), 0.95)
        return self.threshold
    
    def classify(self,x:tokenization_utils_base.BatchEncoding, threshold:float, divergence:Literal["Renyi", "FR"], 
                 scenario:Literal["s0", "s1"], alpha:Optional[float], temperature:float=1):
        anomaly = self.anomaly_score(x, divergence, scenario, alpha, temperature)
        return anomaly>threshold