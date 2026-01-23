import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionDenoiser(nn.Module):
    def __init__(self, dc=64, ds=64, num_treatments=6, t_embed_dim=16, step_embed_dim=32):
        super().__init__()
        self.t_emb = nn.Embedding(num_treatments, t_embed_dim)
        self.step_emb = nn.Embedding(1000, step_embed_dim)  # diffusion steps

        in_dim = dc + ds + t_embed_dim + step_embed_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dc)
        )

    def forward(self, C_noisy, S, t, step):
        t_e = self.t_emb(t)
        s_e = self.step_emb(step)

        x = torch.cat([C_noisy, S, t_e, s_e], dim=1)
        return self.net(x)
