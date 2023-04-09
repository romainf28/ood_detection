from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle
from utils.custom_distilbert import CustomDistilbert
import sys
from tqdm import tqdm
import json


def load_tokenizer_from_checkpoint(checkpoint):
    '''
    Loads tokenizer from a saved chekpoint
    '''
    return AutoTokenizer.from_pretrained(checkpoint)


def load_model_from_checkpoint(checkpoint):
    '''
    Loads model from a saved checkpoint
    '''
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint)

def get_embeddings_from_hidden_states(batch_size, hidden_states):
    '''
    Generates the latent representation of a sequence for each layer from the hidden_states returned by the model
    '''
    n_layers = len(hidden_states)
    latent_representations = []
    latent_representations = torch.zeros((batch_size, model.config.hidden_size,n_layers))
    for i, hidden_state in enumerate(hidden_states):
        latent_representations[:,:,i] = hidden_state[:,0,:].cpu().detach()
    return latent_representations
    

def save_in_ds_train_embeddings(model, tokenizer, in_ds_name='imdb', ds_names={'imdb': 'imdb'}):
    '''
    Save the embeddings of the fine tuned model on the in-distribution dataset
    '''
    dataset = load_dataset(*ds_names[in_ds_name])
    # create data loader
    data_loader = DataLoader(dataset=dataset['train'], batch_size=16, collate_fn=lambda x: [
        el['text'] for el in x])
    in_ds_embeddings = []
    in_ds_logits = []

    model.to('cuda')
    for batch in tqdm(data_loader):
        # tokenize batch
        batch_encoded_input = tokenizer(
            batch, return_tensors='pt', truncation=True, padding=True)['input_ids'].to('cuda')
        outputs = model(batch_encoded_input, output_hidden_states=True)
        in_ds_logits.append(outputs['logits'].cpu().detach())
        
        latent_representations = get_embeddings_from_hidden_states(batch_encoded_input.shape[0], outputs['hidden_states'])
        in_ds_embeddings.append(latent_representations)

    # save embeddings
    with open('./pickle_files/embeddings_inds_train_{}.pkl'.format(in_ds_name), 'wb') as f:
        pickle.dump(in_ds_embeddings, f)
        f.close()
    # save logits
    with open('./pickle_files/logits_inds_train_{}.pkl'.format(in_ds_name), 'wb') as f:
        pickle.dump(in_ds_logits, f)
        f.close()


def save_in_ds_test_embeddings(model, tokenizer, in_ds_name='imdb', ds_names={'imdb': 'imdb'}):
    '''
    Save the embeddings of the fine tuned model on the in-distribution dataset
    '''
    dataset = load_dataset(*ds_names[in_ds_name])
    # create data loader
    data_loader = DataLoader(dataset=dataset['test'], batch_size=16, collate_fn=lambda x: [
        el['text'] for el in x])
    in_ds_embeddings_test = []
    in_ds_logits_test = []

    model.to('cuda')
    for batch in tqdm(data_loader):
        # tokenize batch
        batch_encoded_input = tokenizer(
            batch, return_tensors='pt', truncation=True, padding=True)['input_ids'].to('cuda')
        outputs = model(batch_encoded_input, output_hidden_states=True)
        in_ds_logits_test.append(outputs['logits'].cpu().detach())
        
        latent_representations = get_embeddings_from_hidden_states(batch_encoded_input.shape[0], outputs['hidden_states'])
        in_ds_embeddings_test.append(latent_representations)

    # save embeddings
    with open('./pickle_files/embeddings_inds_test_{}.pkl'.format(in_ds_name), 'wb') as f:
        pickle.dump(in_ds_embeddings_test, f)
        f.close()
    # save logits
    with open('./pickle_files/logits_inds_test_{}.pkl'.format(in_ds_name), 'wb') as f:
        pickle.dump(in_ds_logits_test, f)
        f.close()


def save_ood_embeddings(model, tokenizer, in_ds_name='imdb', ds_names={'imdb': 'imdb'}):
    collate_fns = {'20ng': lambda x : [el['text'].replace('\n', ' ').replace("\\", '') for el in x ],
                  'mnli': lambda x : [[el['text1'],el['text2']] for el in x],
                  'rte' : lambda x : [[el['text1'],el['text2']] for el in x],
                  'wmt16' : lambda x : [el['translation']['en'] for el in x],
                  'multi30k' : lambda x : [el['en'] for el in x],
                  'sst2' : lambda x : [el['sentence'] for el in x],
                  'trec' : lambda x : [el['text'] for el in x]
                  }

    model.to('cuda')

    for ds_name, ds in ds_names.items():
        if ds_name != in_ds_name:
            ood_dataset = load_dataset(*ds)
            data_loader = DataLoader(dataset=ood_dataset['test'],
                                     batch_size=16,
                                     collate_fn=collate_fns[ds_name],
                                     )
            ood_embeddings = []
            ood_logits = []
            for batch in tqdm(data_loader):
                batch_encoded_input = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)[
                    'input_ids'].to('cuda')
                outputs = model(batch_encoded_input,output_hidden_states=True)
                ood_logits.append(outputs['logits'].cpu().detach())
                
                latent_representations = get_embeddings_from_hidden_states(batch_encoded_input.shape[0], outputs['hidden_states'])
                ood_embeddings.append(latent_representations)
                
                

            with open('./pickle_files/embeddings_ood_test_{}.pkl'.format(ds_name), 'wb') as f:
                pickle.dump(ood_embeddings, f)
                f.close()

            with open('./pickle_files/logits_ood_test_{}.pkl'.format(ds_name), 'wb') as f:
                pickle.dump(ood_logits, f)
                f.close()


if __name__ == '__main__':
    with open('datasets_benchmark.json') as json_ds:
        ds_names = json.load(json_ds)
        json_ds.close()

    checkpoint = sys.argv[1]
    in_ds_name = sys.argv[2]
    
    # load tokenizer and model from saved checkpoint
    tokenizer = load_tokenizer_from_checkpoint(checkpoint)
    model = load_model_from_checkpoint(checkpoint)
    
    # Save in-distribution train embeddings
    # save_in_ds_train_embeddings(model, tokenizer, in_ds_name=in_ds_name,
    #                       ds_names=ds_names)

    # Save in-distribution test embeddings
    # save_in_ds_test_embeddings(model, tokenizer, in_ds_name=in_ds_name,
    #                       ds_names=ds_names)

    # save ood embeddings
    save_ood_embeddings(model, tokenizer, in_ds_name=in_ds_name,
                        ds_names=ds_names)
