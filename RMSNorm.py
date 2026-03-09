import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model,
        eps:float=1e-5,
        device:torch.device | None=None,
        dtype:torch.dtype | None=None
    ):
        super().__init__()
        
        self.d_model=d_model
        self.eps=eps
        
        self.weight=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
    
    def _rms(self,x:torch.Tensor)->torch.Tensor:
        return torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        input_dtype=x.dtype
        # 修改类型为float32 防止rms计算溢出
        x=x.to(torch.float32)
        rms=self._rms(x)
        x_normed=x/rms
        return (x_normed*self.weight).to(input_dtype)# 相比之前，没有减mean也没有加偏差，用的不是方差而是rms