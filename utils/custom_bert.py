import torch


class CustomBert(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = len(self.model.bert.encoder.layer)

    def to(self, device):
        self.model.to(device)
        return self

    def forward(self, x, aggregate=True):

        x = self.model.bert.embeddings(x)
        latent_representations = torch.zeros((x.shape[0], x.shape[2], self.n_layers)
                                             )
        for i, layer in enumerate(self.model.bert.encoder.layer):
            x = layer(x)[0]
            # Keeping the CLS tokens
            latent_representations[:, :, i] = x[:, 0, :]
        x_ = self.model.bert.pooler(x)
        x = self.model.classifier(x_)

        return {"embeddings": torch.mean(latent_representations, axis=-1) if aggregate else latent_representations, "logits": x, "attention": x_}
