import torch
from torch import tensor
from typing import Any, Dict, List, Optional
import torch.distributed as dist

__all__= ["AdamW_FourBit_Triton"]

def init_random_generator(gpu, seed = 2020):
    global random_generator
    if random_generator is None:
        random_generator = torch.Generator(device=gpu)
    random_generator.manual_seed(seed)

class AdamW_FourBit_Triton(torch.optim.Optimizer):
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

        # q specific
        if dist.is_initialized():
            seed = torch.randint(1<<31, size=[], device=torch.device('cuda'))
            dist.broadcast(seed, src=0)
            init_random_generator(dist.get_rank(), seed.item()) #avoid stochastic rounding
        
        defaults = dict(
            lr = lr,
            betas=betas,
            eps = eps,
            weight_decay = weight_decay, 
            fused = fused,
        )
        super().__init__(params, defaults)
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("fused", None)
        state_values = list(self.state.values())
        # have to store step as tensor, not scalar
        step_is_tensor = (len(state_values)!=0 and torch.is_tensor(state_values[0]['step']))

        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s["step"]))
    
    def get_subqconfig(self, optimizer_state_name):
        if optimizer_state_name == "exp_avg":
            return self.qconfig.quant.M
        elif optimizer_state_name == "exp_avg_sq":
            return self.qconfig.quant.SQM
        else:
            raise ValueError(f" invalid state name {optimizer_state_name=}")

    def _init_group():
        pass