

import torch
import torch.nn as nn

from models.diffusion.temporal_layers import (
    SinusoidalPE, TimestepEmbedder, TemporalVertexBlock,
)
from models.diffusion.graph_layers import GraphContactLayer


NUM_VERTICES = 778
MOTION_DIM = 99
CONTACT_DIM = NUM_VERTICES
NUM_REGIONS = 6




class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


def grad_scale(x, scale):
    
    return _GradScale.apply(x, scale)




class GeometryInjection(nn.Module):
 

    def __init__(self, d_model, hidden=128, in_scale=10.0, use_anchor=False):
        super().__init__()
        self.use_anchor = use_anchor
        self.in_scale = in_scale                        # hand ~0.1 m -> ~1.0
        in_dim = 3 + (1 if use_anchor else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, verts, anchor=None, valid_mask=None):
        # verts : (B, T, 778, 3)
        rc = (verts - verts.mean(dim=2, keepdim=True)) * self.in_scale
        feat = rc
        if self.use_anchor and anchor is not None:
            d = (verts - anchor.unsqueeze(2)).norm(dim=-1, keepdim=True)
            feat = torch.cat([rc, d], dim=-1)
        g = self.norm(self.mlp(feat))
        if valid_mask is not None:
            g = g * valid_mask[:, :, None, None].float()
        return g


class ContactModule(nn.Module):
  

    def __init__(self, grid_size=16, contact_dim=16):
        super().__init__()
        self.grid_size = grid_size
        self.contact_dim = contact_dim

        self.grid_conv = nn.Sequential(
            nn.Conv3d(3, contact_dim // 4, 3, 1, 1), nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(contact_dim // 4, contact_dim // 2, 3, 1, 1), nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(contact_dim // 2, contact_dim // 2, 3, 1, 1), nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(contact_dim // 2, contact_dim, 3, 1, 1), nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(contact_dim, contact_dim, 3, 1, 1), nn.ReLU(),
        )

        self.contact_conv = nn.Sequential(
            nn.Conv3d(1, contact_dim // 4, 3, 1, 1), nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(contact_dim // 4, contact_dim // 2, 3, 1, 1), nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(contact_dim // 2, contact_dim // 2, 3, 1, 1), nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(contact_dim // 2, contact_dim, 3, 1, 1), nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(contact_dim, contact_dim, 3, 1, 1), nn.ReLU(),
        )

        downsample = sum(
            1 for m in self.contact_conv if isinstance(m, nn.MaxPool3d)
        )
        self.feat_dim = contact_dim * ((grid_size // (2 ** downsample)) ** 3)

        self.unknown_contact = nn.Parameter(
            torch.randn((grid_size, grid_size, grid_size))
        )
        self.unknown_grid = nn.Parameter(
            torch.randn((3, grid_size, grid_size, grid_size))
        )

    def forward(self, c_vol, g_vol, c_mask, g_mask):
        # c_vol : (B, T, N, N, N) | g_vol : (B, N, N, N, 3)
        # c_mask : (B, T)         | g_mask : (B,)
        bz, ts = c_vol.shape[:2]
        N = self.grid_size

        x = c_vol.reshape(bz * ts, N, N, N)
        c_known = c_mask.reshape(bz * ts, 1, 1, 1).expand(bz * ts, N, N, N)
        x = x * c_known + (1 - c_known) * self.unknown_contact.unsqueeze(0)

        g = g_vol.permute(0, 4, 1, 2, 3).contiguous()          # (B, 3, N, N, N)
        g_known = g_mask.view(bz, 1, 1, 1, 1).expand(bz, 3, N, N, N)
        g = g * g_known + (1 - g_known) * self.unknown_grid.unsqueeze(0)

        x = self.contact_conv(x.unsqueeze(1))
        x = x.view(bz, ts, -1)                                  # (B, T, feat_dim)

        g = self.grid_conv(g)
        g = g.view(bz, 1, -1).expand(bz, ts, -1)                # (B, T, feat_dim)
        return x, g


class RegionTemporalBlock(nn.Module):
    """
    Annatomical regions temporal block 

    Input  : (B, T, R, D)
    Output : (B, T, R, D)
    """

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
        B, T, R, D = x.shape

        y = self.norm1(x)
        y = y.permute(0, 2, 3, 1).contiguous().reshape(B * R, D, T)
        y = self.conv(y)
        y = y.reshape(B, R, D, T).permute(0, 3, 1, 2).contiguous()
        x = x + y

        x = x + self.ff(self.norm2(x))

        if valid_mask is not None:
            x = x * valid_mask[:, :, None, None].float()

        return x


class AnatomicalRegionBottleneck(nn.Module):
   

    def __init__(
        self,
        d_model,
        region_map,
        ff_size=1024,
        dropout=0.1,
        n_regions=NUM_REGIONS,
        n_layers=2,
        kernel_size=5,
    ):
        super().__init__()
        self.n_regions = n_regions

        assert region_map is not None, " exige region_map."
        region_map = region_map.detach().long().cpu()

        assert region_map.shape == (NUM_VERTICES,), \
            f"region_map must be ({NUM_VERTICES},), reçu {tuple(region_map.shape)}"

        assert region_map.min() >= 0 and region_map.max() < n_regions, \
            f"region_map [0,{n_regions - 1}]"

        self.register_buffer("region_map", region_map, persistent=True)

        counts = torch.bincount(region_map, minlength=n_regions).float().clamp(min=1.0)
        self.register_buffer("region_counts", counts, persistent=True)

        self.region_emb = nn.Embedding(n_regions, d_model)

        self.region_in = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.region_temporal = nn.ModuleList([
            RegionTemporalBlock(d_model, ff_size, dropout, kernel_size)
            for _ in range(n_layers)
        ])

        self.region_to_vertex = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.gate = nn.Parameter(torch.tensor(0.0))
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, valid_mask=None):
        B, T, V, D = x.shape
        device = x.device
        dtype = x.dtype
        R = self.n_regions

        region_map = self.region_map.to(device)
        counts = self.region_counts.to(device=device, dtype=dtype)

        region_sum = torch.zeros(B, T, R, D, device=device, dtype=dtype)
        idx = region_map.view(1, 1, V, 1).expand(B, T, V, D)

        region_sum.scatter_add_(dim=2, index=idx, src=x)
        region_feat = region_sum / counts.view(1, 1, R, 1)

        r_idx = torch.arange(R, device=device)
        region_feat = region_feat + self.region_emb(r_idx).view(1, 1, R, D)
        region_feat = self.region_in(region_feat)

        if valid_mask is not None:
            region_feat = region_feat * valid_mask[:, :, None, None].float()

        for layer in self.region_temporal:
            region_feat = layer(region_feat, valid_mask=valid_mask)

        region_msg = region_feat[:, :, region_map, :]
        region_msg = self.region_to_vertex(region_msg)

        x = x + torch.tanh(self.gate) * region_msg
        x = self.out_norm(x)

        if valid_mask is not None:
            x = x * valid_mask[:, :, None, None].float()

        return x


class ContactDenoiser(nn.Module):
    MOTION_DIM = MOTION_DIM
    CONTACT_DIM = CONTACT_DIM
    NUM_VERTICES = NUM_VERTICES

    def __init__(
        self,
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        max_len=200,
        video_dim=768,
        text_dim=512,
        contact_grid_dim=16,
        contact_cond_out=256,
        contact_cond_dim=16,
        cond_mask_prob=0.1,
        contact_temporal_layers=2,
        contact_graph_layers=3,
        contact_graph_heads=4,
        contact_kernel_size=5,
        edge_feat_dim=0,
        region_map=None,
        use_region_bottleneck=True,
        region_layers=2,
        num_regions=NUM_REGIONS,
        
        fk=None,                         #  (B,T,99)->(B,T,778,3) = FK de l_motion_pos
        grad_motion_to_contact=0.1,      # contact -> motion (via m_tok + FK), petit
        grad_contact_to_motion=0.1,      # motion  -> contact summary
        geom_hidden=128,
        use_anchor_channel=False,       
        
        detach_motion_to_contact=True,
        detach_contact_to_motion=True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.cond_mask_prob = cond_mask_prob
        self.use_region_bottleneck = use_region_bottleneck
        self.num_regions = num_regions

        self.grad_motion_to_contact = grad_motion_to_contact
        self.grad_contact_to_motion = grad_contact_to_motion

       
        object.__setattr__(self, "_fk", fk)
        self.use_geometry = fk is not None

        if use_region_bottleneck and region_map is None:
            raise ValueError(
                "use_region_bottleneck=True mais region_map=None. "
                "Passe mano_region_map.pt au constructeur, ou mets "
                "use_region_bottleneck=False explicitement."
            )

        pe_max = max(max_len + 100, 1010)

        self.temporal_pe = SinusoidalPE(latent_dim, dropout, max_len=pe_max)
        self.timestep_emb = TimestepEmbedder(latent_dim, self.temporal_pe)

        self.contact_module = ContactModule(
            grid_size=contact_grid_dim,
            contact_dim=contact_cond_dim,
        )

        # text + video + [f_contact ; f_grid] + contact_point(3)
        cond_dim = video_dim + text_dim + self.contact_module.feat_dim * 2 + 3
        self.embed_cond = nn.Linear(cond_dim, latent_dim)

      
        # Motion branch
        
        self.motion_in = nn.Linear(MOTION_DIM, latent_dim)

        motion_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )

        self.motion_transformer = nn.TransformerEncoder(
            motion_layer,
            num_layers=num_layers,
        )

       
        self.motion_pre_head = nn.Linear(latent_dim, MOTION_DIM)

       
        self.motion_out = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, MOTION_DIM),
        )

        # injection 
        self.geom_inject = GeometryInjection(
            latent_dim, hidden=geom_hidden, use_anchor=use_anchor_channel
        )

       
        # Contact branch
        
        self.contact_bit_in = nn.Linear(1, latent_dim)
        self.vertex_emb = nn.Embedding(NUM_VERTICES, latent_dim)
        self.frame_emb = nn.Embedding(pe_max, latent_dim)

        self.motion_to_contact = nn.Linear(latent_dim, latent_dim)
        self.cond_to_contact = nn.Linear(latent_dim, latent_dim)

        self.contact_norm_in = nn.LayerNorm(latent_dim)

        self.contact_temporal_pre = nn.ModuleList([
            TemporalVertexBlock(
                latent_dim,
                ff_size,
                dropout,
                contact_kernel_size,
            )
            for _ in range(contact_temporal_layers)
        ])

        self.contact_graph = nn.ModuleList([
            GraphContactLayer(
                d_model=latent_dim,
                num_heads=contact_graph_heads,
                ff_size=ff_size,
                dropout=dropout,
                edge_feat_dim=edge_feat_dim,
            )
            for _ in range(contact_graph_layers)
        ])

        if use_region_bottleneck:
            self.region_bottleneck = AnatomicalRegionBottleneck(
                d_model=latent_dim,
                region_map=region_map,
                ff_size=ff_size,
                dropout=dropout,
                n_regions=num_regions,
                n_layers=region_layers,
                kernel_size=contact_kernel_size,
            )
        else:
            self.region_bottleneck = None

        self.contact_temporal_post = TemporalVertexBlock(
            latent_dim,
            ff_size,
            dropout,
            contact_kernel_size,
        )

        self.contact_out = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, 1),
        )

        self._init_weights()

        
        nn.init.zeros_(self.motion_out[-1].weight)
        nn.init.zeros_(self.motion_out[-1].bias)

    @property
    def fk(self):
        return self._fk

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _cfg_mask(self, B, device):
        
        if self.training and self.cond_mask_prob > 0.0:
            return torch.bernoulli(
                torch.ones(B, device=device) * self.cond_mask_prob
            ).view(B, 1)

        return torch.zeros(B, 1, device=device)

    def forward(
        self,
        motion_t,        # (B, T, 99)
        contact_t,       # (B, T, 778)
        timesteps,       # (B,)
        video_feats,     # (B, 768)
        text_feats,      # (B, 512)
        contact_vol,     # (B, 16, 16, 16)  
        contact_point,   # (B, 3)
        grid_vol=None,   # (B, 16, 16, 16, 3) 
        contact_mask=None, # (B,) ou (B, T)
        grid_mask=None,  # (B,)
        valid_mask=None, # (B, T)
        edge_index=None,
        edge_features=None,
    ):
        B, T, _ = motion_t.shape
        device = motion_t.device

     
        if video_feats.dim() == 3:
            video_feats = video_feats[:, 0]
        if text_feats.dim() == 3:
            text_feats = text_feats[:, 0]
        if contact_point.dim() == 3:
            contact_point = contact_point[:, 0]

        if contact_vol.dim() == 4:
            c_vol = contact_vol.unsqueeze(1)
        elif contact_vol.dim() == 5:
            c_vol = contact_vol[:, :1]
        else:
            raise ValueError(
                f"contact_vol shape invalid : {tuple(contact_vol.shape)} "
                f"(expected 4D (B,N,N,N) or 5D (B,T,N,N,N))"
            )

        if grid_vol is None:
            raise ValueError(
                
                "(B, N, N, N, 3). Passe-le au forward."
            )

        if contact_mask is None:
            c_mask = torch.ones(B, 1, device=device)
        else:
            c_mask = contact_mask[:, :1] if contact_mask.dim() == 2 \
                     else contact_mask.view(B, 1)
        if grid_mask is None:
            g_mask = torch.ones(B, device=device)
        else:
            g_mask = grid_mask

        contact_feats, grid_feats = self.contact_module(c_vol, grid_vol, c_mask, g_mask)
        contact_feats = torch.cat([contact_feats, grid_feats], dim=-1)
        contact_feats = contact_feats[:, 0]

        cfg_mask = self._cfg_mask(B, device)

        cond_raw = torch.cat([
            video_feats    * (1.0 - cfg_mask),
            text_feats     * (1.0 - cfg_mask),
            contact_feats  * (1.0 - cfg_mask),
            contact_point  * (1.0 - cfg_mask),
        ], dim=-1)

        emb = self.timestep_emb(timesteps)
        cond_emb = self.embed_cond(cond_raw).unsqueeze(0)
        emb = emb + cond_emb

        
        x_m = self.motion_in(motion_t.float()).permute(1, 0, 2)
        x_m = self.temporal_pe(x_m)

        xseq = torch.cat([emb, x_m], dim=0)

        pad_mask = None
        if valid_mask is not None:
            emb_pad = torch.zeros(B, 1, device=device, dtype=torch.bool)
            body_pad = valid_mask == 0
            pad_mask = torch.cat([emb_pad, body_pad], dim=1)

        out = self.motion_transformer(xseq, src_key_padding_mask=pad_mask)
        out_motion = out[1:].permute(1, 0, 2).contiguous()  # (B, T, D)

      
        motion_pre = self.motion_pre_head(out_motion)       # (B, T, 99) = m̂0

        geom_tok = None
        if self.use_geometry:
           
            verts = self._fk(grad_scale(motion_pre, self.grad_motion_to_contact))
            anchor = contact_point if self.geom_inject.use_anchor else None
            geom_tok = self.geom_inject(verts, anchor=anchor, valid_mask=valid_mask)

        
        v_idx = torch.arange(NUM_VERTICES, device=device)
        t_idx = torch.arange(T, device=device)

        bit_tok = self.contact_bit_in(contact_t.float().unsqueeze(-1))
        v_tok = self.vertex_emb(v_idx).view(1, 1, NUM_VERTICES, -1)
        f_tok = self.frame_emb(t_idx).view(1, T, 1, -1)
        m_tok = self.motion_to_contact(
            grad_scale(out_motion, self.grad_motion_to_contact)
        ).unsqueeze(2)
        c_tok = self.cond_to_contact(emb.squeeze(0)).view(B, 1, 1, -1)

        contact_tokens = bit_tok + v_tok + f_tok + m_tok + c_tok
        if geom_tok is not None:
            contact_tokens = contact_tokens + geom_tok      
        contact_tokens = self.contact_norm_in(contact_tokens)

        if valid_mask is not None:
            contact_tokens = contact_tokens * valid_mask[:, :, None, None].float()

       
        for layer in self.contact_temporal_pre:
            contact_tokens = layer(contact_tokens, valid_mask=valid_mask)

        for layer in self.contact_graph:
            contact_tokens = layer(
                contact_tokens,
                edge_index=edge_index,
                edge_features=edge_features,
                valid_mask=valid_mask,
            )

        if self.region_bottleneck is not None:
            contact_tokens = self.region_bottleneck(contact_tokens, valid_mask=valid_mask)

        contact_tokens = self.contact_temporal_post(contact_tokens, valid_mask=valid_mask)

        pred_contact_logits = self.contact_out(contact_tokens).squeeze(-1)

        contact_summary = grad_scale(contact_tokens.mean(dim=2), self.grad_contact_to_motion)
        correction = self.motion_out(torch.cat([out_motion, contact_summary], dim=-1))
        pred_motion = motion_pre + correction               # correction zero-init

        if valid_mask is not None:
            pred_motion = pred_motion * valid_mask[:, :, None].float()
            pred_contact_logits = pred_contact_logits * valid_mask[:, :, None].float()

        return pred_motion, pred_contact_logits