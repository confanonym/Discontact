

import torch


def _bcast(v: torch.Tensor) -> torch.Tensor:
    return v[:, None, None]


def q_sample_contact(c0: torch.Tensor, ab_t: torch.Tensor, p_v: torch.Tensor) -> torch.Tensor:
    """Forward noising per-vertex."""
    ab = _bcast(ab_t).clamp(0.0, 1.0)
    p = p_v[None, None, :].clamp(1e-4, 1.0 - 1e-4)
    prob = ab * c0 + (1.0 - ab) * p
    prob = prob.clamp(0.0, 1.0)
    return torch.bernoulli(prob)


def q_posterior_probs(c_t: torch.Tensor, c0_probs: torch.Tensor,
                      ab_t: torch.Tensor, ab_tm1: torch.Tensor, alpha_t: torch.Tensor,
                      p_v: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    """
    Calcul of p(c_v^{t-1} = 1 | c_v^t), marginalised on c_0 avec c0_probs.
    1. c0_probs : p̂(c_0 = 1) (B, V)
    2. ab_t, ab_tm1, alpha_t : (B, T)
    3. p_v : p_v (V)
    4. c_t : c^t (B, T, V)
   
    6. Return : p(c^{t-1} = 1 | c^t) (B, T, V)
    """
   
    ab_t_b = _bcast(ab_t).clamp(eps, 1.0 - eps)
    ab_tm1_b = _bcast(ab_tm1).clamp(eps, 1.0 - eps)
    alpha_t_b = _bcast(alpha_t).clamp(eps, 1.0 - eps)
    p = p_v[None, None, :].clamp(eps, 1.0 - eps)
    c0_probs = c0_probs.clamp(eps, 1.0 - eps)

    
    q_tm1_1_given_c0_1 = ab_tm1_b + (1.0 - ab_tm1_b) * p
    
    q_tm1_1_given_c0_0 = (1.0 - ab_tm1_b) * p


    q_t_given_tm1_1 = c_t * (alpha_t_b + (1.0 - alpha_t_b) * p) \
                    + (1.0 - c_t) * ((1.0 - alpha_t_b) * (1.0 - p))
    q_t_given_tm1_1 = q_t_given_tm1_1.clamp(eps, 1.0)

    num = q_t_given_tm1_1 * (q_tm1_1_given_c0_1 * c0_probs
                              + q_tm1_1_given_c0_0 * (1.0 - c0_probs))

    
    q_t_1_given_c0_1 = ab_t_b + (1.0 - ab_t_b) * p
    q_t_1_given_c0_0 = (1.0 - ab_t_b) * p
    p_ct_given_c0_1 = c_t * q_t_1_given_c0_1 + (1.0 - c_t) * (1.0 - q_t_1_given_c0_1)
    p_ct_given_c0_0 = c_t * q_t_1_given_c0_0 + (1.0 - c_t) * (1.0 - q_t_1_given_c0_0)
    p_ct_given_c0_1 = p_ct_given_c0_1.clamp(eps, 1.0)
    p_ct_given_c0_0 = p_ct_given_c0_0.clamp(eps, 1.0)

    denom = p_ct_given_c0_1 * c0_probs + p_ct_given_c0_0 * (1.0 - c0_probs)
    denom = denom.clamp(min=eps)

    return (num / denom).clamp(eps, 1.0 - eps)


def kl_bernoulli(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
   
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    return p * (torch.log(p) - torch.log(q)) + \
           (1.0 - p) * (torch.log(1.0 - p) - torch.log(1.0 - q))


def vb_loss_contact(c0_pred_logits: torch.Tensor, c_t: torch.Tensor, c0_gt: torch.Tensor,
                    ab_t: torch.Tensor, ab_tm1: torch.Tensor, alpha_t: torch.Tensor,
                    p_v: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    L_vb = E[ KL(q(c^{t-1} | c^t, c^0) || p_η(c^{t-1} | c^t)) ]
   
    """
    c0_probs = torch.sigmoid(c0_pred_logits)

    # True posterior  c0_gt
    true_p = q_posterior_probs(c_t, c0_gt, ab_t, ab_tm1, alpha_t, p_v)
    # Predicted posterior avec c0_probs
    pred_p = q_posterior_probs(c_t, c0_probs, ab_t, ab_tm1, alpha_t, p_v)

    kl = kl_bernoulli(true_p, pred_p)              # (B, T, V)
    
    kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
    
    kl_per_t = kl.mean(dim=-1)                      # (B, T)
    denom = valid_mask.sum().clamp(min=1.0)
    return (kl_per_t * valid_mask).sum() / denom


@torch.no_grad()
def sample_contact_reverse_step(c_t, c0_pred_logits, ab_t, ab_tm1, alpha_t, p_v, t_idx):
    c0_probs = torch.sigmoid(c0_pred_logits)
    if t_idx == 0:
        return (c0_probs > 0.5).float()
    prob_tm1 = q_posterior_probs(c_t, c0_probs, ab_t, ab_tm1, alpha_t, p_v)
    return torch.bernoulli(prob_tm1)