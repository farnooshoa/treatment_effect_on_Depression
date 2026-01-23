import torch
import torch.nn as nn

class CategoricalEmbedding(nn.Module):
    def __init__(self, num_categories, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embed_dim)

    def forward(self, x):
        return self.embedding(x)


class CategoricalEmbeddingBlock(nn.Module):
    def __init__(self, category_sizes, embed_dim=8):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [CategoricalEmbedding(n, embed_dim) for n in category_sizes]
        )

    def forward(self, x_cat):
        # x_cat shape: [batch_size, num_categorical_features]
        embedded = [
            emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)
        ]
        return torch.cat(embedded, dim=1)
    
class Encoder(nn.Module):
    def __init__(self, input_dim, ds=64, dc=64, dropout=0.1):
        super().__init__()
        self.ds = ds
        self.dc = dc

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.net(x)
        S = h[:, :self.ds]
        C = h[:, self.ds:self.ds + self.dc]
        return S, C
    

class PropensityHead(nn.Module):
    def __init__(self, dc, num_treatments):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dc, 64),
            nn.ReLU(),
            nn.Linear(64, num_treatments)
        )

    def forward(self, C):
        return self.net(C)

class OutcomeModel(nn.Module):
    def __init__(self, ds, dc, num_treatments, treatment_embed_dim=16, dropout=0.1):
        super().__init__()

        self.treatment_embedding = nn.Embedding(num_treatments, treatment_embed_dim)

        input_dim = ds + dc + treatment_embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(self, S, C, t):
        t_emb = self.treatment_embedding(t)
        x = torch.cat([S, C, t_emb], dim=1)
        y_hat = self.net(x)
        return y_hat.squeeze(-1)

class ITEHead(nn.Module):
    def __init__(self, ds, dc, num_treatments):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ds + dc, 128),
            nn.ReLU(),
            nn.Linear(128, num_treatments)
        )

    def forward(self, S, C):
        x = torch.cat([S, C], dim=1)
        return self.net(x)


    
class DiffusionDenoiser(nn.Module):
    def __init__(self, dc, ds, num_treatments,
                 step_embed_dim=32, treatment_embed_dim=16):
        super().__init__()

        self.step_embedding = nn.Embedding(1000, step_embed_dim)
        self.treatment_embedding = nn.Embedding(num_treatments, treatment_embed_dim)

        input_dim = dc + ds + step_embed_dim + treatment_embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dc)
        )

    def forward(self, C_noisy, S, t, step):
        step_emb = self.step_embedding(step)
        t_emb = self.treatment_embedding(t)

        x = torch.cat([C_noisy, S, step_emb, t_emb], dim=1)
        return self.net(x)

if __name__ == "__main__":
    category_sizes = [3, 2, 7, 3, 10]
    batch_size = 4
    num_treatments = 6

    x_num = torch.randn(batch_size, 6)

    x_cat = torch.stack(
        [
            torch.randint(0, size, (batch_size,))
            for size in category_sizes
        ],
        dim=1
    )

    emb = CategoricalEmbeddingBlock(category_sizes)
    x_cat_emb = emb(x_cat)

    x = torch.cat([x_num, x_cat_emb], dim=1)

    encoder = Encoder(input_dim=x.shape[1])
    S, C = encoder(x)

    # Propensity head
    prop_head = PropensityHead(dc=64, num_treatments=num_treatments)
    logits = prop_head(C)
    print("Propensity logits shape:", logits.shape)

    # Outcome model
    outcome_model = OutcomeModel(ds=64, dc=64, num_treatments=num_treatments)
    t = torch.randint(0, num_treatments, (batch_size,))
    y_hat = outcome_model(S, C, t)

    print("Outcome prediction shape:", y_hat.shape)
    ite_head = ITEHead(ds=64, dc=64, num_treatments=num_treatments)
    ite = ite_head(S, C)

    print("ITE output shape:", ite.shape)
        # Diffusion denoiser test
    denoiser = DiffusionDenoiser(
        dc=64,
        ds=64,
        num_treatments=num_treatments
    )

    C_noisy = torch.randn(batch_size, 64)
    step = torch.randint(0, 1000, (batch_size,))
    t = torch.randint(0, num_treatments, (batch_size,))

    C_denoised = denoiser(C_noisy, S, t, step)
    print("Denoised C shape:", C_denoised.shape)
