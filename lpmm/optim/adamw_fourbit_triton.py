import torch
from torch import Tensor
from typing import Any, Dict, List, Optional
import torch.distributed as dist
from dataclasses import dataclass


__all__= ["AdamW_FourBit_Triton"]

@dataclass
class QuantParams:
    bits: int
    group_size: int
    scale_type: str
    quant_type: str
    round_type: str
    signed: bool 
    threshold: int
    enable: bool = True

class FirstMoment(QuantParams):
    bits = 4
    group_size = 128
    scale_type = 'group'
    quant_type = 'nonlinear'
    round_type = 'real-nearest'
    signed = True
    threshold = 4096

class SecondMoment(QuantParams):
    bits = 4
    group_size = 128
    scale_type = 'rank1'
    quant_type = 'power1'
    round_type = 'real-nearest'
    signed = False


def init_random_generator(gpu, seed = 2020):
    global random_generator
    if random_generator is None:
        random_generator = torch.Generator(device=gpu)
    random_generator.manual_seed(seed)

def _get_qenable_fn(p, threshold) -> bool:
    if threshold and p.numel() <= threshold:
        return False
    return True

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
        
        self.config_q_m = FirstMoment()
        self.config_q_sqm = SecondMoment()
        self.qmaps = {}

        defaults = dict(
            lr = lr,
            betas=betas,
            eps = eps,
            weight_decay = weight_decay, 
            fused = fused,
        )
        super().__init__(params, defaults)
    
    def get_qmetadata_by_state_name(self, optimizer_state_name):
        subconfig = self.get_subqconfig(optimizer_state_name)
        md = dict(
            b=subconfig.bits,
            scale_type=subconfig.scale_type,
            quant_type=subconfig.quant_type,
            round_type=subconfig.round_type,
            gp_sz=subconfig.group_size,
            signed=subconfig.signed,
        )
        return md

    def init_qstate(self, p, state_name):
        state = self.state[p]
        field = f"{state_name}_qstate"
        state[field] = {
            "enable": True, 
            "overhead": dict(),
            "qmap": None,
        }
        subconfig = self.get_subqconfig(state_name)
        state[field][
            "enable"
        ] = _get_qenable_fn(p, subconfig.THRESHOLD)
        
        md = self.get_qmetadata_by_state_name(state_name)
        qmap_key = (md['quant_type'], md['b'], md['signed'])
        if qmap_key not in self.qmaps:
            self.qmaps[qmap_key] = create_general_qmap(*qmap_key)
        self.qmaps[qmap_key] = self.qmaps[qmap_key].to(p.device)
        state[field]["qmap"] = self.qmaps[qmap_key]

    
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
            return self.config_q_m
        elif optimizer_state_name == "exp_avg_sq":
            return self.config_q_sqm
        else:
            raise ValueError(f" invalid state name {optimizer_state_name=}")

    def _init_group(
        self,
        group, 
        params_with_grad,
        grads,
        exp_avgs, 
        exp_avgs_sqs,
        state_steps,
        exp_avgs_qmap,
        exp_avgs_sqs_qmap,
    
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("AdamW_FourBit does not support sparse gradients")
            grads.append(p.grad)
            state = self.state[p]

            # lazy init state ------    
            if len(state) ==0:
                state['step'] = torch.tensor(0.0)
                state['exp_avg'] = torch.zeros((), dtype= torch.float, device=p.device)
                self.init_qstate(p,"exp_avg")

                state["exp_avg_sq"] = torch.zeros((), dtype = torch.float, device=p.device)
                self.init_qstate(p, "exp_avg_sq")
            # ------ end state init

            state_steps.append(state["step"])
            exp_avgs.append(state["exp_avg"])
            exp_avgs_sqs.append(state["exp_avg_sq"])

            #exp_avgs_q_enabled.append(self.override_q_enable[id(p)] if id(p) in self.override_q_enable else state["exp_avg_qstate"]["enable"])
            #exp_avg_sqs_q_enabled.append(self.override_q_enable[id(p)] if id(p) in self.override_q_enable else state["exp_avg_sq_qstate"]["enable"])
            #exp_avgs_q_overhead.append(state["exp_avg_qstate"]["overhead"])
            #exp_avg_sqs_q_overhead.append(state["exp_avg_sq_qstate"]["overhead"])
            exp_avgs_qmap.append(state["exp_avg_qstate"]["qmap"])
            # exp_avg_sqs_qmap.append(state["exp_avg_sq_qstate"]["qmap"])

    @torch.no_grad()
    def step(self, closure=None):
        """ single optimization step 
        """

        loss = None
        if closure:
            with torch.enable_grad:
                loss = closure() 
        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            exp_avgs_qmap = []
            exp_avg_sqs_qmap = []
            
            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                exp_avgs_qmap,
                exp_avg_sqs_qmap,
            ) 

            kwargs = dict(
                params_with_grad = params_with_grad,
                grads = grads,
                exp_avgs = exp_avgs,
                exp_avg_sqs = exp_avg_sqs,
                state_steps = state_steps,
                exp_avgs_qmap=exp_avgs_qmap,
                exp_avg_sqs_qmap = exp_avg_sqs_qmap,
                beta1 = beta1,
                beta2= beta2,
                lr = group['lr'],
                weight_decay=group['weight_decay'],
                eps = group['eps']

            )

            _single_tensor_step(**kwargs)
            

def _single_tensor_step(
        params_with_grad: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        exp_avgs_qmap: List,
        exp_avg_sqs_qmap: List,
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float
        
):
    for i, param in enumerate(params_with_grad):
        grad = grads[i]
        q_exp_avg = exp_avgs[i]
        q_exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        step_t +=1

        # decoupled weight decay
        param.mul_(1 - lr * weight_decay)

        # dequant
        q_enabled = True # todo 
        sq_enabled = True

        if q_exp_avg.numel() < 2:
            q_exp_avg.data = exp_avg = torch.zeros_like(param, memory_format=torch.preserve_format)
        elif q_enabled:
            exp_avg = avgs_dequant(q_exp_avg, shape = param.shape)
        else:
            exp_avg = q_exp_avg
        
        if q_exp_avg_sq.numel() < 2:
            q_exp_avg_sq.data = exp_avg_sq = torch.zeros_like(param, memory_format = torch.preserve_format)
        elif sq_enabled:
            exp_avg_sq = sqs_dequant(q_exp_avg_sq, shape = param.shape,  )
        else:
            exp_avg_sq = q_exp_avg_sq

        # update avgs
        exp_avg.lerp_(grad, 1-beta1)
        exp_avg_sq.mul_(beta2).addcumul_(grad, grad, value = 1-beta2)

        step = step_t.item()
        bias_corr1 = 1-beta1** step
        bias_corr2 = 1 - beta2 **step
        step_size = lr / bias_corr1
        bias_corr2_sqrt = bias_corr2.sqrt()

        denom = (exp_avg_sq.sqrt() / bias_corr2_sqrt).add_(eps)
        # weight update
        param.addcdiv_(exp_avg, denom, value =-step_size)

        # quantize
        qx, gen = avgs_quant(exp_avg, shape = param.shape)
        q_exp_avg.data = qx

        qx, gen = sqs_quant(exp_avg_sq, shape = param.shape)


def avgs_quant(x, shape):
    """ quantize the exp_avg 

    """
    group_size = 128

    qx = x.detach()

    meta = {}
    meta['dtype'] = x.dtype
    meta['stride'] = x.stride()

    # quant scaling for exp_avgs
    qx = group_tensor(x, group_size)
    max_per_row = max_reduce_except_dim(qx.abs(), 0)
    qx = qx.div(max_per_row)
    scaled_shape = qx.shape

    # metadata = max_per_row, scaled_shape

    qx = nonlinear_quant(qx)

    return qx, metadata


def nonlinear_quant(qx):
    grouped_qx = group_tensor(qx, 2048)
    res = cuda_kernel_pack_nonlinear(grouped_qx)

    





def group_tensor(x: Tensor, group_size: int):
    """ break the tensor into rows of 'group size', padding if needed with zeros"""
    x_flat = x.flatten()
    num_flat = x_flat.shape[0]

    # reshape
    if num_flat % group_size != 0:
        # pad
        new_num_flat = (num_flat // group_size +1) * group_size
        delta = new_num_flat - num_flat
        
        x_flat = torch.cat([x_flat, torch.zeros([delta], dtype = x.dtype, device = x.device)], dim=0)
    x_groups = x_flat.view(-1, group_size)
    return x_groups

def max_reduce_except_dim(input, dim):
    """ compute max along all dims except provided dim """ 
    rank = input.dim() 
    result = input
    if rank:
        assert dim < rank, f"reducing tensor with {rank} dimensions failed"
        for d in range(rank):
            if d != dim:
                result = result.max(dim=d, keepdim=True).values
    return result


