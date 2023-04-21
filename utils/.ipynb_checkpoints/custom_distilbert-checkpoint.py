import torch

class CustomDistilbert(torch.nn.Module):

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.n_layers = len(self.base_model.distilbert.transformer.layer)

    def to(self, device):
        self.base_model.to(device)
        return self

    def forward(self, x,attention_mask, aggregate=True):
        
        x = self.base_model.distilbert.embeddings(x)
        latent_representations = torch.zeros((x.shape[0], x.shape[2],self.n_layers)
        )
        for i, layer in enumerate(self.base_model.distilbert.transformer.layer) :
            x = layer(x,attention_mask)[0]
            latent_representations[:,:,i] = x[:,0,:] # Keeping the CLS tokens
        pooled_repr = x[:, 0, :]  # get the last hidden state of the CLS token
        x = self.base_model.classifier(pooled_repr)
                
        return {"embeddings" : torch.mean(latent_representations, axis=-1) if aggregate else latent_representations, "logits": x}
        