

import numpy as np
import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.shape[0]])


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, pe: SinusoidalPE):
        super().__init__()
        self.pe = pe
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, timesteps):
        t_emb = self.pe.pe[timesteps].squeeze(1)
        return self.time_embed(t_emb).unsqueeze(0)


class ContactVolumeEncoder(nn.Module):
    
    def __init__(self, contact_dim=16, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, contact_dim // 4, 3, 1, 1), nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(contact_dim // 4, contact_dim // 2, 3, 1, 1), nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(contact_dim // 2, contact_dim, 3, 1, 1), nn.ReLU(),
            nn.MaxPool3d(2, 2),
        )
        self.proj = nn.Linear(contact_dim * (2 ** 3), out_dim)

    def forward(self, vol):
        x = self.conv(vol.unsqueeze(1))
        return self.proj(x.flatten(1))


class TemporalVertexBlock(nn.Module):
   
    
    def __init__(self, d_model, ff_size=1024, dropout=0.1, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, valid_mask=None):
        B, T, V, D = x.shape

        y = self.norm1(x)
        y = y.permute(0, 2, 3, 1).contiguous().reshape(B * V, D, T)
        y = self.conv(y)
        y = y.reshape(B, V, D, T).permute(0, 3, 1, 2).contiguous()
        x = x + y
        x = x + self.ff(self.norm2(x))

        if valid_mask is not None:
            x = x * valid_mask[:, :, None, None].float()
        return x