
import torch
import torch.nn as nn
from tqdm import tqdm

from models.diffusion.denoiser import NUM_VERTICES
from models.diffusion.schedules import (
    cosine_schedule_continuous, cosine_schedule_discrete, make_alpha_bars,
)
from models.diffusion.contact_d3pm import (
    q_sample_contact, sample_contact_reverse_step,
)
from models.diffusion.utils import aa_to_rot6d, rot6d_to_aa

import common.rotation_conversions as rot


def fk_verts_and_joints(motion_61d, mano, device):
    
    B, T, _ = motion_61d.shape
    m = motion_61d.reshape(B * T, 61)

    theta = m[:, :48]
    cam_t = m[:, 58:61]
    theta = torch.nan_to_num(theta, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0)
    betas = torch.zeros(B * T, 10, device=device, dtype=m.dtype)

    orient_mat = rot.axis_angle_to_matrix(theta[:, :3]).unsqueeze(1)
    pose_mat   = rot.axis_angle_to_matrix(theta[:, 3:].reshape(-1, 15, 3))

    out    = mano(global_orient=orient_mat, hand_pose=pose_mat, betas=betas, pose2rot=False)
    joints = (out.joints   + cam_t[:, None, :]).reshape(B, T, -1, 3)
    verts  = (out.vertices + cam_t[:, None, :]).reshape(B, T, 778, 3)

    return joints, verts


class HybridDDPM(nn.Module):

    def __init__(
        self,
        denoiser,
        T: int = 1000,
        vertex_marginals: torch.Tensor = None,
        mano=None,          
        **kwargs,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.T = T
        self.mano = mano   

        # === Motion : DDPM Gaussien (cosine) ===
        betas_m   = cosine_schedule_continuous(T)
        alphas_m  = 1.0 - betas_m
        ab_m      = make_alpha_bars(betas_m)
        ab_m_prev = torch.cat([torch.tensor([1.0]), ab_m[:-1]])
        post_var  = betas_m * (1.0 - ab_m_prev) / (1.0 - ab_m).clamp(min=1e-8)
        post_lv   = torch.log(torch.cat([post_var[1:2], post_var[1:]]).clamp(min=1e-20))

        self.register_buffer('betas_m',   betas_m)
        self.register_buffer('alphas_m',  alphas_m)
        self.register_buffer('ab_m',      ab_m)
        self.register_buffer('ab_m_prev', ab_m_prev)
        self.register_buffer('post_lv_m', post_lv)

        # === Contact : D3PM cosine ===
        betas_c     = cosine_schedule_discrete(T)
        alphas_c    = (1.0 - betas_c).clamp(0.0, 1.0)
        log_alpha_c = torch.log(alphas_c.clamp(min=1e-20))
        ab_c        = torch.exp(torch.cumsum(log_alpha_c, dim=0))
        ab_c_prev   = torch.cat([torch.tensor([1.0]), ab_c[:-1]])

        self.register_buffer('betas_c',   betas_c)
        self.register_buffer('alphas_c',  alphas_c)
        self.register_buffer('ab_c',      ab_c)
        self.register_buffer('ab_c_prev', ab_c_prev)

        
        if vertex_marginals is None:
            vertex_marginals = torch.full((NUM_VERTICES,), 0.05)
            print('[HybridDDPM] WARNING: vertex_marginals=None, fallback uniform 0.05.')
        assert vertex_marginals.shape == (NUM_VERTICES,)
        vertex_marginals = vertex_marginals.clamp(1e-3, 1.0 - 1e-3)
        self.register_buffer('p_v', vertex_marginals)

   
    # Forward noising

    def q_sample_motion(self, m0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ab  = self.ab_m[t][:, None, None]
        eps = torch.randn_like(m0)
        return torch.sqrt(ab) * m0 + torch.sqrt(1.0 - ab) * eps


    # Training step
   

    def forward(self, batch: dict, loss_fn, edge_index=None, edge_features=None) -> dict:
        device = batch['motion'].device
        B      = batch['motion'].shape[0]

        motion      = batch['motion'].float()           # (B, T, 61) axis-angle
        contact_map = batch['contact_map'].float()      # (B, T, 778)
        valid_mask  = batch['valid_mask'].float()       # (B, T)
        video_feats = batch['video_feats'][:, 0].float()
        text_feats  = batch['text_feats'][:, 0].float()
        contact_vol = batch['contact'][:, 0].float()
        contact_pt  = batch['contact_point'][:, 0].float()
        grid_vol    = batch['grid'].float()             # (B, 16, 16, 16, 3)
        grid_mask   = batch['grid_mask'].float()        # (B,)
        contact_mask = batch['contact_mask'].float()   # (B, T)

        t = torch.randint(1, self.T, (B,), device=device)

        
        motion_6d   = aa_to_rot6d(motion)               # (B, T, 99)
        motion_t_6d = self.q_sample_motion(motion_6d, t)
        contact_t   = q_sample_contact(contact_map, self.ab_c[t], self.p_v)

        pred_motion_6d, pred_contact_logits = self.denoiser(
            motion_t_6d, contact_t, t,
            video_feats, text_feats, contact_vol, contact_pt,
            grid_vol=grid_vol, contact_mask=contact_mask, grid_mask=grid_mask,
            valid_mask=valid_mask, edge_index=edge_index, edge_features=edge_features,
        )

       
        joints_gt = joints_pred = None
        if self.mano is not None:
            joints_gt, _   = fk_verts_and_joints(motion,                   self.mano, device)
            pred_aa        = rot6d_to_aa(pred_motion_6d)
            joints_pred, _ = fk_verts_and_joints(pred_aa,                  self.mano, device)

        losses = loss_fn(
            motion_pred         = pred_motion_6d,
            motion_gt           = motion_6d,
            contact_pred_logits = pred_contact_logits,
            contact_t           = contact_t,
            contact_gt          = contact_map,
            valid_mask          = valid_mask,
            ab_c_t              = self.ab_c[t],
            ab_c_tm1            = self.ab_c_prev[t],
            alpha_c_t           = self.alphas_c[t],
            p_v                 = self.p_v,
            joints_pred         = joints_pred,
            joints_gt           = joints_gt,
        )
        return losses

    
    # Sampling
   

    @torch.no_grad()
    def sample(
        self,
        video_feats,
        text_feats,
        contact_vol,
        contact_point,
        T_seq,
        device,
        grid_vol=None,
        contact_mask=None,
        grid_mask=None,
        verbose=True,
        edge_index=None,
        edge_features=None,
        T_diff_steps=None,
        unconditional=False,
    ) -> dict:
       
        B = video_feats.shape[0]

        if grid_vol is None:
            N        = contact_vol.shape[-1]
            grid_vol = torch.zeros(B, N, N, N, 3, device=device)
            grid_mask = torch.zeros(B, device=device)
        if grid_mask is None:
            grid_mask = torch.ones(B, device=device)
        if contact_mask is None:
            contact_mask = torch.ones(B, 1, device=device)

        if unconditional:
            video_feats   = torch.zeros_like(video_feats)
            text_feats    = torch.zeros_like(text_feats)
            contact_vol   = torch.zeros_like(contact_vol)
            contact_point = torch.zeros_like(contact_point)
            grid_vol      = torch.zeros_like(grid_vol)
            grid_mask     = torch.zeros_like(grid_mask)
            contact_mask  = torch.zeros_like(contact_mask)

        m_6d         = torch.randn(B, T_seq, 99, device=device)
        p_v_expanded = self.p_v[None, None, :].expand(B, T_seq, -1)
        contact_t    = torch.bernoulli(p_v_expanded)
        valid_mask   = torch.ones(B, T_seq, device=device)

        if T_diff_steps is None or T_diff_steps >= self.T:
            timesteps = list(reversed(range(self.T)))
        else:
            step      = max(1, self.T // T_diff_steps)
            timesteps = list(reversed(range(0, self.T, step)))

        iterator = tqdm(timesteps, total=len(timesteps), desc='Hybrid sampling') \
                   if verbose else timesteps

        m0_pred_6d, contact_logits = None, None

        for t_idx in iterator:
            t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)

            m0_pred_6d, contact_logits = self.denoiser(
                m_6d, contact_t, t_tensor,
                video_feats, text_feats, contact_vol, contact_point,
                grid_vol=grid_vol, contact_mask=contact_mask, grid_mask=grid_mask,
                valid_mask=valid_mask, edge_index=edge_index, edge_features=edge_features,
            )

            # Reverse motion (DDPM)
            beta_m   = self.betas_m[t_idx]
            alpha_m  = self.alphas_m[t_idx]
            ab_m_t   = self.ab_m[t_idx]
            ab_m_prv = self.ab_m_prev[t_idx]
            lv_m     = self.post_lv_m[t_idx]

            coef_x0 = beta_m * torch.sqrt(ab_m_prv) / (1.0 - ab_m_t).clamp(min=1e-8)
            coef_xt = (1.0 - ab_m_prv) * torch.sqrt(alpha_m) / (1.0 - ab_m_t).clamp(min=1e-8)
            mu_m    = coef_x0 * m0_pred_6d + coef_xt * m_6d

            noise_m = torch.zeros_like(m_6d) if t_idx == 0 else torch.randn_like(m_6d)
            m_6d    = mu_m + torch.exp(0.5 * lv_m) * noise_m

            # Reverse contact (D3PM)
            contact_t = sample_contact_reverse_step(
                c_t            = contact_t,
                c0_pred_logits = contact_logits,
                ab_t           = self.ab_c[t_tensor],
                ab_tm1         = self.ab_c_prev[t_tensor],
                alpha_t        = self.alphas_c[t_tensor],
                p_v            = self.p_v,
                t_idx          = t_idx,
            )

        return {
            'motion':         rot6d_to_aa(m_6d),
            'motion_x0':      rot6d_to_aa(m0_pred_6d),
            'contact_logits': contact_logits,
            'contact':        contact_t,
        }