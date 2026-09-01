

import torch
import torch.nn.functional as F

from models.diffusion.contact_d3pm import vb_loss_contact


NUM_REGIONS = 6



def masked_mean(loss_t, mask):
    return (loss_t * mask).sum() / mask.sum().clamp(min=1.0)


def masked_mean_vt(loss_vt, mask):
    return masked_mean(loss_vt.mean(dim=-1), mask)



# Motion


def l_motion(motion_pred, motion_gt, mask):
    loss = F.mse_loss(motion_pred, motion_gt, reduction='none').mean(dim=-1)
    return masked_mean(loss, mask)


def l_motion_pos(joints_pred, joints_gt, mask):
    if joints_pred.dim() == 4:
        joints_pred = joints_pred.reshape(joints_pred.shape[0], joints_pred.shape[1], -1)
        joints_gt   = joints_gt.reshape(joints_gt.shape[0], joints_gt.shape[1], -1)
    loss = F.mse_loss(joints_pred, joints_gt, reduction='none').mean(dim=-1)
    return masked_mean(loss, mask)




def l_contact_aux(logits, gt, mask):
   
    loss = F.binary_cross_entropy_with_logits(
        logits,
        gt.float(),
        reduction='none',
    )
    return masked_mean_vt(loss, mask)


def _region_density(C, region_map, n_regions=NUM_REGIONS):
  
    B, T, V = C.shape
    device = C.device
    region_map = region_map.to(device)
    R = torch.zeros(B, T, n_regions, device=device, dtype=C.dtype)
    for r in range(n_regions):
        idx = (region_map == r)
        if idx.sum() > 0:
            R[:, :, r] = C[:, :, idx].mean(dim=-1)
    return R


def l_region_density(logits, gt, mask, region_map):
    
    prob   = torch.sigmoid(logits)
    d_pred = _region_density(prob,       region_map)
    d_gt   = _region_density(gt.float(), region_map)
    loss   = F.mse_loss(d_pred, d_gt, reduction='none').mean(dim=-1)
    return masked_mean(loss, mask)




def compute_all_losses(
    motion_pred,
    contact_pred_logits,
    contact_t,
    motion_gt,
    contact_gt,
    valid_mask,
    ab_c_t,
    ab_c_tm1,
    alpha_c_t,
    p_v,
    joints_pred=None,
    joints_gt=None,
    region_map=None,
    **kwargs,
):
    device = contact_pred_logits.device
    losses = {}

    losses['l_motion'] = l_motion(motion_pred, motion_gt, valid_mask)

    losses['l_motion_pos'] = l_motion_pos(joints_pred, joints_gt, valid_mask) \
        if joints_pred is not None and joints_gt is not None \
        else torch.tensor(0.0, device=device)

    losses['l_contact_vb'] = vb_loss_contact(
        c0_pred_logits=contact_pred_logits,
        c_t=contact_t,
        c0_gt=contact_gt,
        ab_t=ab_c_t,
        ab_tm1=ab_c_tm1,
        alpha_t=alpha_c_t,
        p_v=p_v,
        valid_mask=valid_mask,
    )

    losses['l_contact_aux'] = l_contact_aux(
        contact_pred_logits,
        contact_gt,
        valid_mask,
    )

    losses['l_region_density'] = l_region_density(
        contact_pred_logits,
        contact_gt,
        valid_mask,
        region_map=region_map,
    ) if region_map is not None else torch.tensor(0.0, device=device)

    losses['total'] = sum(losses.values())

    return losses
region_density = _region_density