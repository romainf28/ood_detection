import numpy as np
from typing import Optional, Literal
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, tokenization_utils_base, AutoTokenizer
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt


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

def define_threshold(tensor_score:torch.tensor, r:float):
    '''
    Return the wanted threshold if we want a fix recall
    '''
    return torch.quantile(tensor_score, 1-r)
    
def auroc(tensor_score_in:torch.tensor, tensor_score_out:torch.tensor, plot:bool=False, name=""):
    '''
    Compute the auroc, the best f1 and the threshold for the best f1
    Plot the curve of auroc if wanted
    '''
    r_min = min(torch.min(tensor_score_out), torch.min(tensor_score_in))
    r_max = max(torch.max(tensor_score_out), torch.max(tensor_score_in))
    tpr = []
    fpr = []
    best_f1 = 0
    best_thresh = 0
    y_true = torch.cat([torch.tensor([1]).expand(tensor_score_out.shape[0]).cpu(), 
                        torch.tensor([0]).expand(tensor_score_in.shape[0]).cpu()])
    for threshold in torch.linspace(r_min, r_max, 50,):
        y_pred_out = torch.round((tensor_score_out>=threshold).float(), decimals=0).cpu()
        y_pred_in = torch.round((tensor_score_in>=threshold).float(), decimals=0).cpu()
        tpr.append(y_pred_out.mean())
        fpr.append(y_pred_in.mean())
        y_pred = torch.cat([y_pred_out,y_pred_in])
        f1 = f1_score(y_true, y_pred)
        if f1_score(y_true, y_pred)>best_f1:
            best_f1 = f1
            best_thresh = threshold
    auc = -1 * np.trapz(tpr, fpr)
    if plot:
        plt.plot(fpr,tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(name + ' ROC curve, AUC = %.2f'%auc)
        plt.legend(loc="lower right")
        plt.savefig("./Images/"+name + '_AUC.png')
        plt.show()
    return auc, best_f1, best_thresh

def show_gaphs(tensor_score_in:torch.tensor, tensor_score_out:torch.tensor, name=""):
    '''
    Show all usefull graphs for our problem
    '''
    auc, f1, thresh = auroc(tensor_score_in, tensor_score_out, True, name = name)
    print(f1, thresh)
    
    plt.hist(tensor_score_out.cpu(),bins=50, alpha = 0.5,density=True,label="out")
    plt.hist(tensor_score_in.cpu(),bins=50, alpha = 0.5,density=True,label="in")
    plt.title(name + '_hist')
    plt.legend()
    plt.savefig("./Images/" + name+ "_hist")
    plt.show()

class GenerationModel :
    '''
    Class that permits to compute the anomaly score of a model with a given input
    '''
    def __init__(self, model_name:str):

        if torch.cuda.is_available():
            self.device = 'cuda'
            
        else:
            self.device = 'cpu'
        print(self.device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


        self.softmax = torch.nn.Softmax(dim=2).to(self.device)

        self.vocab_size = self.model.config.vocab_size
        self.threshold = None
        self.bag = None

    def generate_set(self, x:tokenization_utils_base.BatchEncoding, temperature:float=1):
        '''
        Generate the set of proba for a bathce of input and return the mask of the sequence generated
        We need it because within a batch the outputs size is fixed to the max output size
        '''
        outputs = self.model.generate(**x.to(self.device), return_dict_in_generate=True,
                                      output_scores=True, num_beams=1)
        output_scores = torch.stack(list(outputs.scores), dim=0).permute(1,0,2)
        sequences_mask = (outputs.sequences[:,1:] != self.model.config.pad_token_id).float()
        del outputs
        return (self.softmax(output_scores/temperature) * (sequences_mask.unsqueeze(-1))).to(self.device), sequences_mask
    
    def anomaly_score(self,x:tokenization_utils_base.BatchEncoding, divergence:Literal["Renyi", "FR"], 
                      scenario:Literal["s0", "s1"], alpha:Optional[float], temperature:float=1):
        '''
        Compute the anomaly score for a batch with the wanted parameters
        '''
        set_proba, sequences_mask = self.generate_set(x.to(self.device), temperature=temperature)
        
        if scenario=="s0":
            proba = torch.ones(set_proba.shape)/self.vocab_size
        

            proba = proba.to(self.device)
            if divergence == "Renyi":
                if not alpha:
                    raise AttributeError("When you use Renyi divergence, you must define an alpha")
                div = renyi_divergence(set_proba, proba, alpha)
                div[div==torch.inf]=0
                div = torch.sum(div, dim=1) / torch.sum(sequences_mask, dim=1)
            elif divergence =="FR":
                div = FR_distance(set_proba, proba)
                div[div==torch.inf]=0
                div = torch.sum(div, dim=1) / torch.sum(sequences_mask, dim=1)
            else: 
                raise AttributeError("The divergence should be 'Renyi' or 'FR'")
            del proba
            del set_proba
            del sequences_mask
            return div
        
        elif scenario=="s1":
            if self.bag==None:
                raise AttributeError("Please generate a bag of distribution to compare with")
            
            bag_x = torch.sum(set_proba, dim=1) / torch.sum(sequences_mask, dim=1).unsqueeze(1)
            del sequences_mask
            del set_proba
            div = torch.tensor([np.inf]).expand(bag_x.shape[0]).to(self.device)
            batch_bag_size = 64
            for i in range(self.bag.shape[0]//batch_bag_size):
                
                bag_prob_extanded = self.bag[i*batch_bag_size:min((i+1)*batch_bag_size, self.bag.shape[0]),:]\
                    .unsqueeze(0).expand(bag_x.shape[0], -1, self.bag.shape[1]).to(self.device)

                bag_x_extanded = bag_x.unsqueeze(1).expand(bag_x.shape[0], bag_prob_extanded.shape[1], 
                                                           bag_x.shape[1]).to(self.device)
                if divergence == "Renyi":
                    if not alpha:
                        raise AttributeError("When you use Renyi divergence, you must define an alpha")
                    div = torch.min(torch.cat([renyi_divergence(bag_x_extanded, bag_prob_extanded, alpha), div.unsqueeze(1)], 
                                              dim=1), dim=1).values
                elif divergence =="FR":
                    div = torch.min(torch.cat([FR_distance(bag_x_extanded, bag_prob_extanded), div.unsqueeze(1)], 
                                              dim=1), dim=1).values
                else: 
                    raise AttributeError("The divergence should be 'Renyi' or 'FR'")
                del bag_x_extanded, bag_prob_extanded
            del bag_x
            return div
            

    def anomaly_score_loader(self, data_loader:DataLoader, divergence:Literal["Renyi", "FR"], 
                         scenario:Literal["s0", "s1"], alpha:Optional[float], temperature:float=1):
        '''
        Compute the anomaly score for a dataloader of batches with the wanted parameters
        '''
        list_anomaly = []
        for _, batch in tqdm(enumerate(data_loader), desc="Computing anomaly score for each data", 
                             total=data_loader.__len__()):
            x = self.tokenizer(batch["sourceString"], return_tensors="pt", 
                               padding=True).to(self.device)
            anomaly = self.anomaly_score(x, divergence, scenario, alpha, temperature)
            del x
            list_anomaly.append(anomaly)
        return torch.cat(list_anomaly)
    
    
    
    def classify(self,x:tokenization_utils_base.BatchEncoding, threshold:float, 
                 divergence:Literal["Renyi", "FR"], scenario:Literal["s0", "s1"], 
                 alpha:Optional[float], temperature:float=1):
        '''
        Classify an batch of input with the wanted parameters and a given threshold
        '''
        anomaly = self.anomaly_score(x, divergence, scenario, alpha, temperature)
        return anomaly>threshold
    
    def generate_bag(self, data_loader:DataLoader, temperature:float=1):
        '''
        Generate the bag of output distribution given a data loader of input.
        '''
        list_proba = []
        for _, batch in tqdm(enumerate(data_loader), desc="Generate bag of distribution", 
                             total=data_loader.__len__()):
            del _
            x = self.tokenizer(batch["sourceString"], return_tensors="pt", 
                               padding=True).to(self.device)
            set_proba, sequences_mask = self.generate_set(x.to(self.device), temperature=temperature)
            list_proba.append(torch.sum(set_proba, dim=1) / torch.sum(sequences_mask, dim=1).unsqueeze(1))
            del sequences_mask
            del set_proba
        del x
        del batch
        self.bag = torch.cat(list_proba)
        del list_proba
        return self.bag