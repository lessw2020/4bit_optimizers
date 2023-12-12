import torch
from torch import tensor
from typing import List, Optional


__all__= ["FourBit_AdamW_Triton"]

class FourBit_AdamW_Triton(torch.optim.Optimizer):
    """ 4bit AdamW with Triton fusion
    based on lpmm 4bit Optimizers """

    def __init__(
            self,
            params, 
            lr = 1e-3,
            betas = (0.9, 0.999), 
            eps = 1e-8,
            weight_decay = 1e-2, 
            *,
            fused: Optional[bool] = False,
    ):
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr=}")
        if not 0.0 < eps:
            raise ValueError(f"Invalid eps value: {eps=}")
        if not 0.0 < betas[0] < 1.0:
            raise ValueError(f"Invalid Beta[0]: {betas[0]=}")
        if not 0.0 < betas[1] < 1.0:
            raise ValueError(f"Invalid Beta[1]: {betas[1]=}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay=}")

        defaults = dict(
            lr = lr,
            betas=betas,
            eps = eps,
            weight_decay = weight_decay, 
            fused = fused,
        )
        super().__init__(params, defaults)