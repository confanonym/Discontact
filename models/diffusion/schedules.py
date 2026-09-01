

import math
import numpy as np
import torch


def cosine_schedule_continuous(T: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Cosine schedule DDPM (Nichol & Dhariwal 2021)."""
    betas = []
    for i in range(T):
        t1, t2 = i / T, (i + 1) / T
        f = lambda t: math.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
        betas.append(min(1.0 - f(t2) / f(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def cosine_schedule_discrete(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule discret (DiGress / Hoogeboom et al.)."""
    steps = T + 2
    x = np.linspace(0, steps, steps)
    ab = np.cos(0.5 * np.pi * ((x / steps) + s) / (1.0 + s)) ** 2
    ab = ab / ab[0]
    alphas = ab[1:] / ab[:-1]
    betas = 1.0 - alphas
    return torch.from_numpy(betas[:T]).float()


def make_alpha_bars(betas: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(1.0 - betas, dim=0)