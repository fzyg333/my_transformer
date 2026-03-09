import torch
import torch.nn as nn
import einops

class Linear(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        device:torch.device | None=None,
        dtype:torch.dtype | None=None,
        bias:bool=False
    ):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.weight=nn.Parameter(torch.empty((in_dim,out_dim),device=device,dtype=dtype))
        self.bias=nn.Parameter(torch.empty(out_dim,device=device,dtype=dtype)) if bias else None
        self._init_weight()
        
    def forward(self,x):
        o=x@self.weight
        if self.bias is not None:
            o=o+self.bias
        return o
    
    def _init_weight(self):
        mean=0.0
        std=1.0/(2*(self.in_dim+self.out_dim)**0.5)
        nn.init.trunc_normal_(self.weight,mean=mean,std=std,a=-3*std,b=3*std)

class RoPEEmbedding(nn.Module):
    def __init__(
        self,
        theta:float,
        d_k:int,
        max_seq_len:int,
        device:torch.device | None=None
    ):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        inv_freq=1.0/(theta**(torch.arange(0,d_k,2,device=device,dtype=torch.float32)/d_k))
        self.register_buffer("inv_freq",inv_freq,persistent=False)
    
    def _rotate_half(self,x):
        x=einops.rearrange(x, "... (d j) -> ... d j", j=2)
        x1,x2=x.unbind(dim=-1)
        return einops.rearrange(torch.stack((-x2,x1),dim=-1),"... d j-> ... (d j)")
    
    def forward(self,x:torch.Tensor,token_pos:int | None=None)->torch.Tensor:
        if token_pos is None:   #token_pos本来就不需要给出作为参数
            seq_len=x.shape[-2]
            token_pos=torch.arange(seq_len,device=x.device)
            token_pos=token_pos.unsqueeze(0)
        
        theta=torch.einsum("...i , j -> ... i j",token_pos,self.inv_freq)
        cos=torch.cos(theta).repeat_interleave(2,dim=-1)
        sin=torch.sin(theta).repeat_interleave(2,dim=-1)
        x_rotated=(x*cos)+(self._rotate_half(x)*sin)
        return x_rotated

def stable_softmax(
    x:torch.Tensor,
    dim:int=-1
)->torch.Tensor:
    max=torch.max(x,dim=dim,keepdim=True).values
    exp=torch.exp(x-max)
    sum_exp=torch.sum(exp,dim=dim,keepdim=True)
    softmax=exp/sum_exp
    return softmax

def qkv_atn_calculate(
    q:torch.Tensor,
    k:torch.Tensor,
    v:torch.Tensor,
    mask:torch.Tensor | None=None,
)->torch.Tensor:
    d_k=q.size(-1)
    scores=torch.matmul(q,k.transpose(-2,-1))/(d_k**0.5)
    
    if mask is not None:
        scores=scores.masked_fill(mask==0,float("-inf"))
    
    attn_weights=stable_softmax(scores,dim=-1)
    output=torch.matmul(attn_weights,v)
    return output

class MHA(nn.Module):
    def __init__(
        self,
        d_model:int,
        num_heads:int,
        use_rope:bool=True,
        theta:float=10000.0,
        max_seq_len:int=2048,
        device:torch.device | None=None,
        dtype:torch.dtype | None=None
    ):
        super().__init__()
        assert d_model%num_heads==0,"d_model不能整除num_heads"
        
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        
        self.q_linear=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_linear=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_linear=Linear(d_model,d_model,device=device,dtype=dtype)
        self.out_linear=Linear(d_model,d_model,device=device,dtype=dtype)
        
        self.use_rope=use_rope
        if use_rope:
            self.rope=RoPEEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device
            )
            
    def _create_causal_mask(self,seq_len:int,device:torch.device)->torch.Tensor:
        mask=torch.tril(torch.ones(seq_len,seq_len,device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        x:torch.Tensor,
        token_pos:torch.Tensor | None=None
    )->torch.Tensor:
        batch_size,seq_len,_=x.size()
        causal_mask=self._create_causal_mask(seq_len,x.device)
        Q=self.q_linear(x).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        K=self.k_linear(x).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        V=self.v_linear(x).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        
        if self.use_rope:
            Q,K=self.rope(Q,token_pos),self.rope(K,token_pos)
            
        attn_output=qkv_atn_calculate(Q,K,V,mask=causal_mask)
        attn_output=attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        output=self.out_linear(attn_output)
        return output

