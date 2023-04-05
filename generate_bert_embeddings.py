from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle
from custom_bert import CustomBert
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


def save_in_ds_embeddings(model, tokenizer, in_ds_name='imdb', ds_names={'imdb': 'imdb'}, aggregate=True):
    '''
    Save the embeddings of the fine tuned model on the in-distribution dataset
    '''
    dataset = load_dataset(ds_names[in_ds_name])
    # create data loader
    data_loader = DataLoader(dataset=dataset['test'], batch_size=8, collate_fn=lambda x: [
        el['text'] for el in x])
    in_ds_embeddings = []

    custom_bert = CustomBert(model).to('cuda')
    for batch in tqdm(data_loader):
        # tokenize batch
        batch_encoded_input = tokenizer(
            batch, return_tensors='pt', truncation=True, padding=True)['input_ids'].to('cuda')
        # forward pass of the custom bert model
        outputs = custom_bert.forward(batch_encoded_input, aggregate=aggregate)
        in_ds_embeddings.append(outputs['embeddings'].cpu().detach())

    # save embeddings
    with open('./pickle_files/embeddings_inds_test_{}.pkl'.format(in_ds_name), 'wb') as f:
        pickle.dump(in_ds_embeddings, f)


def save_ood_embeddings(model, tokenizer, in_ds_name='imdb', ds_names={'imdb': 'imdb'}, aggregate=True):
    batch_sizes = {}
    collate_fns = {}

    custom_bert = CustomBert(model).to('cuda')

    for ds_name, ds in ds_names.items():
        if ds_name != in_ds_name:
            ood_dataset = load_dataset(*ds)
            data_loader = DataLoader(dataset=ood_dataset['test'],
                                     batch_size=batch_sizes[ds_name],
                                     collate_fn=collate_fns[ds_name],
                                     )
            ood_embeddings = []
            for batch in tqdm(data_loader):
                tk_batch = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)[
                    'input_ids'].to('cuda')
                outputs = custom_bert.forward(tk_batch, aggregated=False)
                ood_embeddings.append(outputs['embeddings'].cpu().detach())

    with open('.pickle_files/embeddings_ood_test_{}.pkl'.format(ds_name), 'wb') as f:
        pickle.dump(ood_embeddings, f)


if __name__ == '__main__':
    with open('datasets_benchmark.json') as json_ds:
        ds_names = json.load(json_ds)
        json_ds.close()

    checkpoint = sys.argv[1]
    in_ds_name = sys.argv[2]
    aggregate = sys.argv[3]
    # load tokenizer and model from saved checkpoint
    tokenizer = load_tokenizer_from_checkpoint(checkpoint)
    model = load_model_from_checkpoint(checkpoint)

    # Save in-distribution embeddings
    save_in_ds_embeddings(model, tokenizer, in_ds_name=in_ds_name,
                          ds_names=ds_names, aggregate=aggregate)

    # save ood embeddings
    save_ood_embeddings(model, tokenizer, in_ds_name=in_ds_name,
                        ds_names=ds_names, aggregate=aggregate)
