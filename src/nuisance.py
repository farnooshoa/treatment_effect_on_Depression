import torch
import torch.nn as nn
import torch.nn.functional as F

class NuisanceXEncoder(nn.Module):
    """Encodes x_num + x_cat embeddings into a hidden vector."""
    def __init__(self, category_sizes, num_numeric=6, cat_embed_dim=8, hidden=128, dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, cat_embed_dim) for n in category_sizes])
        in_dim = num_numeric + cat_embed_dim * len(category_sizes)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_num, x_cat):
        cat_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        cat_emb = torch.cat(cat_emb, dim=1)
        x = torch.cat([x_num, cat_emb], dim=1)
        return self.net(x)

class PropensityNuisance(nn.Module):
    def __init__(self, x_encoder, hidden=128, num_treatments=6):
        super().__init__()
        self.x_encoder = x_encoder
        self.head = nn.Linear(hidden, num_treatments)

    def forward(self, x_num, x_cat):
        h = self.x_encoder(x_num, x_cat)
        return self.head(h)  # logits

class OutcomeNuisance(nn.Module):
    def __init__(self, x_encoder, hidden=128, num_treatments=6, t_embed_dim=16, dropout=0.1):
        super().__init__()
        self.x_encoder = x_encoder
        self.t_emb = nn.Embedding(num_treatments, t_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden + t_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_num, x_cat, t):
        h = self.x_encoder(x_num, x_cat)
        te = self.t_emb(t)
        y = self.net(torch.cat([h, te], dim=1))
        return y.squeeze(-1)
