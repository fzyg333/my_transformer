import torch
import torch.nn as nn
import einops
from MHA import Linear
import torch.nn.functional as F

def silu(x:torch.Tensor)->torch.Tensor:
    return x*torch.sigmoid(x)

class Router(nn.Module):
    def __init__(self,d_model:int,num_experts:int):
        super().__init__()
        
        self.d_model=d_model
        self.num_experts=num_experts
        self.linear=Linear(d_model,num_experts)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.linear(x)

class Expert(nn.Module):
    def __init__(self,d_model:int,d_ff:int,device:torch.device | None=None,dtype:torch.dtype | None=None):
        super().__init__()
        
        self.up=Linear(d_model,d_ff,device=device,dtype=dtype)
        self.down=Linear(d_ff,d_model,device=device,dtype=dtype)
        self.gate=Linear(d_model,d_ff,device=device,dtype=dtype)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.down(silu(self.up(x))*self.gate(x))

class MOE(nn.Module):
    def __init__(
        self,
        d_model:int,
        d_ff:int,
        num_experts:int,
        top_k:int=1,
        router_jitter:float=0.0,#路由器扰动
        z_loss_coef:float=1e-3,
        lb_loss_coef:float=1e-1,
        device:torch.device | None=None,
        dtype:torch.dtype | None=None,
    ):
        super().__init__()
        
        self.router=Router(d_model,num_experts)
        self.experts=nn.ModuleList([Expert(d_model,d_ff,device=device,dtype=dtype) for _ in range(num_experts)])
        self.num_experts=num_experts
        self.top_k=top_k
        self.router_jitter=router_jitter
        self.z_loss_coef=z_loss_coef
        self.lb_loss_coef=lb_loss_coef
    
    @staticmethod
    def _z_loss(logits:torch.Tensor)->torch.Tensor:
        log_sum_exp=torch.logsumexp(logits,dim=-1)
        z_loss=torch.mean(log_sum_exp**2)
        return z_loss
    
    @staticmethod
    def _load_balance_loss(
        router_probs:torch.Tensor,
        topk_indices:torch.Tensor,
        num_experts:int
    )->torch.Tensor:
        p=router_probs.mean(dim=(0,1))
        dispatch=F.one_hot(topk_indices,num_classes=num_experts).to(router_probs.dtype)
        f=dispatch.mean(dim=(0,1,2))
        return num_experts*torch.sum(p*f)
    
    def forward(self,x:torch.Tensor)->dict[str,torch.Tensor]:
        batch_size,seq_len,d_model=x.size()
        logits=self.router(x)
        
        if self.router_jitter >0.0 and self.training:
            noise=torch.randn_like(logits) * self.router_jitter
            logits=logits+noise
        
        z_loss=self._z_loss(logits)
        router_probs=torch.softmax(logits,dim=-1)
        
        topk_logits,topk_indices=torch.topk(logits,self.top_k,dim=-1)
        if self.top_k==1:
            topk_gates=router_probs.gather(-1,topk_indices)
        else:
            topk_gates=torch.softmax(topk_logits,dim=-1)
        lb_loss=self._load_balance_loss(router_probs,topk_indices,self.num_experts)
        
        #将batch_size展开
        x_flat=x.reshape(batch_size*seq_len,d_model)
        out_flat=x_flat.new_zeros((batch_size*seq_len,d_model))
        
        if self.top_k==1:
            #每个token对应top1的id和得分
            expert_ids=topk_indices.reshape(batch_size*seq_len)
            expert_gates=topk_gates.reshape(batch_size*seq_len)
            
            for e in range(self.num_experts):
                pos=(expert_ids==e).nonzero(as_tuple=False).squeeze(1)
                if pos.numel()==0:
                    continue
                
                x_expert=x_flat.index_select(0,pos)
                y_expert=self.experts[e](x_expert)
                y_expert=y_expert*expert_gates.index_select(0,pos).unsqueeze(1)
                
                out_flat.index_add_(0,pos,y_expert)
            
            counts=torch.bincount(expert_ids,minlength=self.num_experts).to(x.dtype)
            tokens_per_expert=counts/(batch_size*seq_len)
        else:
            token_ids=torch.arange(batch_size*seq_len,device=x.device).unsqueeze(1).expand(batch_size*seq_len,self.top_k).reshape(-1)
            expert_ids=topk_indices.reshape(-1)
            expert_gates=topk_gates.reshape(-1)
            
            for e in range(self.num_experts):
                pos=(expert_ids==e).nonzero(as_tuple=False).squeeze(1)
                if pos.numel==0:
                    continue
                
                tok=token_ids.index_select(0,pos)
                x_expert=x_flat.index_select(0,tok)
                y_expert=self.experts[e](x_expert)
                y_expert=y_expert*expert_gates.index_select(0,pos).unsqueeze(1)
                
                out_flat.index_add_(0,tok,y_expert)
            
            counts=torch.bincount(expert_ids,minlength=self.num_experts).to(x.dtype)
            tokens_per_expert=counts/(batch_size*seq_len*self.top_k)
        
        expert_outputs=out_flat.view(batch_size,seq_len,d_model)
        
        return {
            "output":expert_outputs,
            "tokens_per_expert":tokens_per_expert,
            "z_loss":z_loss,
            "z_loss_scaled":z_loss*self.z_loss_coef,
            "lb_loss":lb_loss,
            "lb_loss_scaled":lb_loss*self.lb_loss_coef
        }