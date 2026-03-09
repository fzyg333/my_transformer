import math
import torch
from typing import Iterable,Optional

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params:Iterable[torch.nn.Parameter],
        lr:float=1e-3,#学习率
        betas:tuple[float,float]=(0.9,0.999),#beta=0代表只使用当前的梯度
        eps:float=1e-8,
        weight_decay:float=1e-2,
    ):
        if lr<0.0:
            raise ValueError("学习率不能为负")
        if eps<=0.0:
            raise ValueError("epsilon值不能为负")
        if weight_decay<0.0:
            raise ValueError("weight_decay不能为负")
        if not isinstance(betas,tuple) or len(betas)!=2:
            raise ValueError("betas值错误")
        beta1,beta2=betas
        if not (0.0<=beta1<1.0):
            raise ValueError("beta1值错误")
        if not (0.0<=beta2<1.0):
            raise ValueError("beta2值错误")
        
        defaults=dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        super().__init__(params,defaults)
    
    @torch.no_grad()
    def step(self,closure:Optional[callable]=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():
                loss=closure()
        
        for group in self.param_groups:
            lr:float=group["lr"]
            beta1,beta2=group["betas"]
            eps:float=group["eps"]
            weight_decay:float=group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad=p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW不支持稀疏矩阵梯度的更新")
                
                state=self.state[p]
                if len(state)==0:
                    state["step"]=0
                    state["exp_avg"]=torch.zeros_like(p,memory_format=torch.preserve_format)
                    state["exp_avg_sq"]=torch.zeros_like(p,memory_format=torch.preserve_format)
                
                exp_avg:torch.Tensor=state["exp_avg"]
                exp_avg_sq:torch.Tensor=state["exp_avg_sq"]
                state["step"]+=1
                t=state["step"]
                
                exp_avg.mul_(beta1).add_(grad,alpha=(1.0-beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value=(1.0-beta2))
                
                bias_correction1=1.0-beta1**t
                bias_correction2=1.0-beta2**t
                
                step_size=lr/bias_correction1
                
                denom=(exp_avg_sq/bias_correction2).sqrt().add_(eps)
                
                if weight_decay!=0.0:#权重衰减
                    p.mul_(1.0-lr*weight_decay)
                
                p.addcdiv_(exp_avg,denom,value=-step_size)
                
        return loss
    
@torch.no_grad()
def gradient_clip(
    parameters:Iterable[torch.nn.Parameter],
    max_l2_norm:float,
    eps:float=1e-6
)->None:
    
    total_norm=0.0
    for p in parameters:
        if p.grad is not None:
            param_norm=p.grad.data.norm(2)#注意:norm函数将被弃用
            total_norm+=param_norm.item()**2
    
    total_norm=total_norm**0.5
    
    clip_coef=max_l2_norm/(total_norm+eps)
    if clip_coef<1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

def cosine_annealing_lr(
    step:int,
    lr_max:float,
    lr_min:float,
    Tw:int,
    Tc:int
)->float:
    if Tw>0 and step<Tw:
        return (step/Tw)*lr_max#step=0时返回值为0,可以改为step+1/Tw
    
    if step<=Tc:
        if Tc==Tw:
            return lr_max
        
        progress=(step-Tw)/(Tc-Tw)
        return lr_min+0.5*(1.0+math.cos(math.pi*progress))*(lr_max-lr_min)
    
    return lr_min