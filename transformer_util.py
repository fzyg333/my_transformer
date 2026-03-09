from dataclasses import dataclass
import json
from typing import Any
from dataclasses import fields,field
import torch
import random
import numpy as np
import torch.nn as nn
from MHA import MHA,Linear
from FFN import FFN
from RMSNorm import RMSNorm
from tokenizer_util import load_tokenizer_from_dir

def print_color(content: str, color: str = "green"):
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
    }
    reset = "\033[0m"
    print(f"{colors.get(color, '')}{content}{reset}")

@dataclass
class ModelConfig:
    vocab_size:int=350#注意要与tokenizer保持一致
    max_seq_len:int=256
    d_model:int=512
    d_ff:int=1344
    num_heads:int=16
    num_layers:int=4
    dropout:float=0.1
    use_rms_norm:bool=True
    pre_norm:bool=True
    eos_token_id:int=256
    device=torch.device("cuda")#不应该加.type
    
    use_rope:bool=True
    rope_theta:float=10000.0
    
    use_moe:bool=False
    num_experts:int=4
    top_k:int=1
    router_jitter: float = 0.1
    z_loss_coef: float = 1e-3
    lb_loss_coef: float = 1e-1
    
    tie_weights:bool=False
    use_final_norm:bool=False
    
    @classmethod
    def from_json(cls,json_path:str)->"ModelConfig":
        with open(json_path,"r",encoding="utf-8") as f:
            data=json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls,data:dict[str,Any])->"ModelConfig":
        allowed={f.name for f in fields(cls)}
        filtered:dict[str,Any]={k:v for k,v in data.items() if k in allowed}
        return cls(**filtered)
    
    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
    
    def to_json(self,path:str):
        with open(path,"w",encoding="utf-8") as f:
            json.dump(self.to_dict(),f,ensure_ascii=False,indent=2)
    
@dataclass
class TrainConfig:
    batch_size:int=256
    num_steps:int=100
    dataset_dir:str="result"
    train_data_path:str="result/train.bin"
    eval_data_path:str="result/eval.bin"
    # 优化器相关参数
    betas: tuple = field(default=(0.9, 0.98))
    weight_decay: float = 1e-5
    max_lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 5
    max_grad_norm: float = 1.0
    # 日志相关参数
    wandb_logging: bool = True
    eval_log_interval: int = 5
    sampling_log_interval: int = 20
    
    model_name:str="my_transformer"
    save_checkpoint_dir:str="checkpoints"
    device=torch.device("cuda")
    debug_mode: bool = False
    use_mixed_precision: bool = True
    log_moe_every: int = 5
    #随机数种子
    seed: int = 2026
    
    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        with open(path,"r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls,data:dict[str,Any])->"TrainConfig":
        data=dict(data)
        if "betas" in data:
            b=data["betas"]
            if isinstance(b,list):
                b=tuple(b)
            data["betas"]=(float(b[0]),float(b[1]))
            
        return cls(**data)
    
    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self, path: str) -> None:
        with open(path,"w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

def seed_everything(seed:int)->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TransformerBlock(nn.Module):
    def __init__(self,config:ModelConfig):
        super().__init__()
        self.config=config
        self.mha=MHA(
            d_model=config.d_model,
            num_heads=config.num_heads,
            use_rope=config.use_rope,#每个MHA层都要再RoPE一次？
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.device
        )
        self.use_moe=config.use_moe
        if self.use_moe:
            from MOE import MOE
            self.ffn=MOE(
                d_model=config.d_model,
                d_ff=config.d_ff,
                num_experts=config.num_experts,
                top_k=config.top_k,
                router_jitter=config.router_jitter,
                z_loss_coef=config.z_loss_coef,
                lb_loss_coef=config.lb_loss_coef,
                device=config.device
            )
        else:
            self.ffn=FFN(
                d_model=config.d_model,
                d_ff=config.d_ff,
                device=config.device
            )
        self.norm1=RMSNorm(config.d_model,eps=1e-5,device=config.device)
        self.norm2=RMSNorm(config.d_model,eps=1e-5,device=config.device)

    def forward(self,x:torch.Tensor,token_pos:torch.Tensor | None=None)->torch.Tensor | tuple:
        aux={
            "z_loss":x.new_zeros(()),
            "z_loss_scaled":x.new_zeros(()),
            "tokens_per_expert":None,
            "lb_loss":x.new_zeros(()),
            "lb_loss_scaled":x.new_zeros(())
        }
        
        x=x+self.mha(self.norm1(x),token_pos=token_pos)# 没有dropout

        if self.use_moe:
            out=self.ffn(self.norm2(x))
            x=x+out["output"]
            
            aux["tokens_per_expert"]=out.get("tokens_per_expert",None)
            aux["z_loss"] = out.get("z_loss", x.new_zeros(()))
            aux["z_loss_scaled"] = out.get("z_loss_scaled", x.new_zeros(()))
            aux["lb_loss"] = out.get("lb_loss", x.new_zeros(()))
            aux["lb_loss_scaled"] = out.get("lb_loss_scaled", x.new_zeros(()))
        else:
            x=x+self.ffn(self.norm2(x))
        
        return x,aux

class OutputLayer(nn.Module):
    def __init__(self,d_model,vocab_size,use_norm:bool=False):
        super().__init__()
        self.linear=Linear(d_model,vocab_size)
        self.norm=RMSNorm(d_model) if use_norm else nn.Identity()
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.linear(self.norm(x))
        
class TransformerLM(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.config=config
        self.token_embeddinng=nn.Embedding(config.vocab_size,config.d_model)
        self.layers=nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])#应该可以分别控制每个层是否使用NOE
        self.final_norm=RMSNorm(config.d_model,eps=1e-5,device=config.device)
        self.output_layer=OutputLayer(config.d_model,config.vocab_size,use_norm=config.use_final_norm)
        
        if config.tie_weights:
            self.output_layer.linear.weight=self.token_embeddinng.weight.T#应该是要加转置，但原代码中没有
    
    def forward(self,x:torch.Tensor,token_pos:torch.Tensor | None=None)->tuple[torch.Tensor,dict]:#修改了输出类型指定
        x=self.token_embeddinng(x)
        total_z_loss_scaled=x.new_zeros(())
        tokens_per_expert_all=[]
        total_lb_loss_scaled=x.new_zeros(())
        moe_layers=0
        
        for layer in self.layers:
            x,aux=layer(x,token_pos=token_pos)
            if self.config.use_moe:
                total_z_loss_scaled=total_z_loss_scaled+aux["z_loss_scaled"]
                total_lb_loss_scaled=total_lb_loss_scaled+aux["lb_loss_scaled"]
                tokens_per_expert_all.append(aux["tokens_per_expert"])
                moe_layers+=1
        
        x=self.final_norm(x)
        logits=self.output_layer(x)
        
        aux_out = {
            "z_loss_scaled": total_z_loss_scaled,
            "moe_layers": moe_layers,
            "tokens_per_expert": tokens_per_expert_all,  # list[Tensor] or []
            "lb_loss_scaled": total_lb_loss_scaled,
        }
        
        return logits,aux_out

