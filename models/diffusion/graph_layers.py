

import torch
import torch.nn as nn


class SparseMeshAttention(nn.Module):
    
    
    def __init__(self, d_model, num_heads=4, edge_feat_dim=0, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5
        self.edge_feat_dim = edge_feat_dim

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if edge_feat_dim > 0:
            self.edge_film_gamma = nn.Linear(edge_feat_dim, num_heads)
            self.edge_film_beta = nn.Linear(edge_feat_dim, num_heads)

    def forward(self, x, edge_index, edge_features=None):
        """
        x             : (B, T, V, D)
        edge_index    : (2, E) long
        edge_features : (E, edge_feat_dim) 
        
        Return : (B, T, V, D)
        """
        B, T, V, D = x.shape
        E = edge_index.shape[1]
        H, dh = self.num_heads, self.d_head

        q = self.q_proj(x).view(B, T, V, H, dh)
        k = self.k_proj(x).view(B, T, V, H, dh)
        v = self.v_proj(x).view(B, T, V, H, dh)

        src = edge_index[0].long().to(x.device)
        dst = edge_index[1].long().to(x.device)

        q_dst = q[:, :, dst, :, :]                     # (B, T, E, H, dh)
        k_src = k[:, :, src, :, :]
        v_src = v[:, :, src, :, :]

        scores = (q_dst * k_src).sum(dim=-1) * self.scale  # (B, T, E, H)

        if edge_features is not None and self.edge_feat_dim > 0:
            gamma = self.edge_film_gamma(edge_features)
            beta = self.edge_film_beta(edge_features)
            scores = scores * (1.0 + gamma[None, None, :, :]) + beta[None, None, :, :]

        scores_exp = self._segment_softmax(scores, dst, V)
        scores_exp = self.dropout(scores_exp)

        weighted_v = scores_exp.unsqueeze(-1) * v_src  # (B, T, E, H, dh)

        out = torch.zeros(B, T, V, H, dh, device=x.device, dtype=x.dtype)
        dst_idx = dst.view(1, 1, E, 1, 1).expand(B, T, E, H, dh)
        out.scatter_add_(dim=2, index=dst_idx, src=weighted_v)

        return self.out_proj(out.reshape(B, T, V, D))

    @staticmethod
    def _segment_softmax(scores, dst, V):
       
        B, T, E, H = scores.shape
        dst_exp = dst.view(1, 1, E, 1).expand(B, T, E, H)

        
        max_per_dst = torch.full((B, T, V, H), -float('inf'),
                                  device=scores.device, dtype=scores.dtype)
        max_per_dst.scatter_reduce_(dim=2, index=dst_exp, src=scores,
                                     reduce='amax', include_self=True)
        scores_centered = scores - max_per_dst.gather(dim=2, index=dst_exp)

       
        scores_exp = torch.exp(scores_centered)
        sum_per_dst = torch.zeros(B, T, V, H, device=scores.device, dtype=scores.dtype)
        sum_per_dst.scatter_add_(dim=2, index=dst_exp, src=scores_exp)
        norm = sum_per_dst.gather(dim=2, index=dst_exp).clamp(min=1e-8)

        return scores_exp / norm


class GraphContactLayer(nn.Module):
    
    
    def __init__(self, d_model, num_heads=4, ff_size=1024, dropout=0.1, edge_feat_dim=0):
        super().__init__()
        self.attn = SparseMeshAttention(
            d_model=d_model, num_heads=num_heads,
            edge_feat_dim=edge_feat_dim, dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_index=None, edge_features=None, valid_mask=None):
        if edge_index is None:
            return x

        attn_out = self.attn(self.norm1(x), edge_index, edge_features)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.norm2(x))

        if valid_mask is not None:
            x = x * valid_mask[:, :, None, None].float()
        return x