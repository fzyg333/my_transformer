import torch
import numpy as np
from tokenizer_util import load_tokenizer_from_dir
from transformer_util import TrainConfig
from contextlib import nullcontext
from dataclasses import dataclass
from adamw import gradient_clip,cosine_annealing_lr
from transformer_util import TransformerLM
import gc
from tqdm import trange
import os
from generator import generate
import wandb

@dataclass
class State:
    pos:int=0

def clear_memory()->None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

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

def save_checkpoint(
    model:TransformerLM,
    optimizer:torch.optim.Optimizer,
    iteration,
    out_path:str,
)->None:
    state={
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "iteration":iteration,
    }
    torch.save(state,out_path)
    
    print(f"checkpoint save to {out_path}")





def get_ctx(use_mix_precision:bool,device:torch.device,verbose:bool=True):
    if use_mix_precision and device.type=="cuda":
        if verbose:
            print("使用混合精度")
        return torch.autocast(device_type="cuda",dtype=torch.bfloat16)#使用bf16混合精度
    else:
        if verbose:
            print("不使用混合精度")
        return nullcontext()

def data_loading_sequential(
    x:np.ndarray | torch.Tensor,
    batch_size:int,
    context_length:int,#==seq_len
    device: torch.device,
    state:State,
    *,
    stride:int | None=None# ==seq_len
)->tuple[torch.Tensor,torch.Tensor]:
    # if stride is None:
    #     stride=context_length
    
    n=x.numel()
    if n-context_length-1<0:
        raise ValueError("错误:数据序列太短")
    
    #原代码在此处似乎过于繁琐  
    end=state.pos+batch_size*context_length+1
    if end>n:               #如果剩下的数据不足以支持一次训练，则从头开始读取数据
        state.pos=0
        end=batch_size*context_length+1
    
    base=x[state.pos:end]
    #这是个生成模型
    inputs=base.as_strided(size=(batch_size,context_length),stride=(context_length,1))
    targets=base[1:].as_strided(size=(batch_size,context_length),stride=(context_length,1))
    state.pos+=batch_size*context_length

    if device.type=="cuda":
        inputs=inputs.to(device,non_blocking=True).long()
        targets=targets.to(device,non_blocking=True).long()
    else:
        inputs=inputs.long().to(device)
        targets=targets.long().to(device)
    
    return inputs,targets

def cross_entropy(logits:torch.Tensor,targets:torch.Tensor):
    logits=logits-torch.max(logits,dim=1,keepdim=True).values
    log_probs=logits-torch.log(torch.sum(torch.exp(logits),dim=1,keepdim=True))

    targets=targets.unsqueeze(1)
    loss=log_probs.gather(1,targets).squeeze(1)
    loss=-loss.mean()
    return loss

def perplexity(loss:torch.Tensor)->torch.Tensor:
    return torch.exp(loss)

def eval_model(
    model:TransformerLM,
    train_config:TrainConfig
):
    model.eval()
    eval_loss=0.0
    eval_perplexity=0.0
    #加载评估数据库
    original_data=np.memmap(
        train_config.eval_data_path,
        dtype=np.uint16,
        mode="r+",
    )
    x=torch.from_numpy(original_data)
    total_tokens=len(original_data)
    num_eval_batches=total_tokens//(train_config.batch_size*model.config.max_seq_len)
    
    state=State(pos=0)
    with torch.no_grad():
        for _ in trange(num_eval_batches):
            inputs,targets=data_loading_sequential(
                x=x,
                batch_size=train_config.batch_size,
                context_length=model.config.max_seq_len,
                device=train_config.device,
                state=state,
            )
            
            logits,aux=model(inputs)
            logits=logits.view(-1,logits.size(-1))
            targets=targets.view(-1)
            loss=cross_entropy(logits,targets)
            eval_loss+=loss.item()
            eval_perplexity+=perplexity(loss).item()
    
    eval_loss=torch.tensor(eval_loss/num_eval_batches)
    eval_perplexity=torch.tensor(eval_perplexity/num_eval_batches)
    
    model.train()
    
    return eval_loss,eval_perplexity

def train(model:TransformerLM,optimizer:torch.optim.Optimizer,train_config:TrainConfig):
    tokenizer=load_tokenizer_from_dir(train_config.dataset_dir)
    
    original_data=np.memmap(
        train_config.train_data_path,
        dtype=np.uint16,
        mode="r+",
    )
    x=torch.from_numpy(original_data)
    
    best_eval_loss=float("inf")
    ctx=get_ctx(train_config.use_mixed_precision,train_config.device)
    
    #训练循环
    state=State(pos=0)
    for step in range(train_config.num_steps):
        log_dict={}
        
        inputs,targets=data_loading_sequential(
            x=x,
            batch_size=train_config.batch_size,
            context_length=model.config.max_seq_len,
            device=train_config.device,
            state=state
        )
        
        #计算loss
        with ctx:
            logits,aux=model(inputs)
            logits=logits.view(-1,logits.size(-1))#将最终得分展开
            targets=targets.view(-1)
            loss=cross_entropy(logits,targets)
            
            if model.config.use_moe:
                z_loss_scaled=aux["z_loss_scaled"]
                moe_layers=aux["moe_layers"]
                lb_loss_scaled=aux["lb_loss_scaled"]
                
                loss=loss+(z_loss_scaled/moe_layers)+(lb_loss_scaled/moe_layers)
        
        #反向传播
        optimizer.zero_grad(set_to_none=True)#梯度清0
        loss.backward()#计算梯度
        gradient_clip(model.parameters(),max_l2_norm=train_config.max_grad_norm)#限制梯度不超过max_l2_norm
        
        #调整学习率
        lr=cosine_annealing_lr(
            step=step,
            lr_max=train_config.max_lr,
            lr_min=train_config.min_lr,
            Tw=train_config.warmup_steps,
            Tc=train_config.num_steps-train_config.warmup_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"]=lr
        optimizer.step()
        
        #日志打印
        if train_config.wandb_logging:
            log_dict["train/loss"]=loss.item()
            log_dict["train/perplexity"]=perplexity(loss).item()
            log_dict["train/lr"]=lr
        
        print_color(
            f"Step{step+1}/{train_config.num_steps},loss:{loss.item()},lr:{lr}","green"
        )
        if model.config.use_moe:
            tokens_per_expert=aux["tokens_per_expert"]
            if step%train_config.log_moe_every==0:
                layers_to_log=sorted(set([0,model.config.num_layers//2,model.config.num_layers-1]))
                for layer_idx in layers_to_log:
                    tpe=tokens_per_expert[layer_idx].detach().float().cpu().numpy()
                    msg=" | ".join([f"专家{e}处理tokens数:{tpe[e]:.3f}" for e in range(len(tpe))])
                    print_color(f"[step{step}] Layer {layer_idx} toekns_per_expert:{msg}","magenta")
                    if train_config.wandb_logging:
                        for e in range(len(tpe)):
                            log_dict[f"moe/layer_{layer_idx}_expert_{e}_tokens"]=tpe[e]
        #评估模型
        if train_config.eval_log_interval>0 and (step+1)%train_config.eval_log_interval==0:
            # 清理内存
            del inputs,targets,logits,loss
            clear_memory()
            
            print_color("评估模型...","blue")
            eval_loss,eval_perplexity=eval_model(model,train_config)
            if train_config.wandb_logging:
                log_dict["eval/loss"]=eval_loss.item()
                log_dict["eval/perplexity"]=eval_perplexity.item()
                
            print_color(f"评估loss:{eval_loss.item()},评估困惑度:{eval_perplexity.item()}","blue")
            if eval_loss.item()<best_eval_loss:#这里应该加上.item()
                best_eval_loss=eval_loss.item()
                print_color(f"新的最低loss:{best_eval_loss}")
                out_path=os.path.join(
                    train_config.save_checkpoint_dir,
                    train_config.model_name,
                    f"best_model_step_{step+1}.pt",
                )
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=step+1,
                    out_path=out_path,
                )
        
        # 产生样例
        if train_config.sampling_log_interval>0 and (step+1)%train_config.sampling_log_interval==0:
            generated_outputs=generate(
                model=model,
                prompt="Once upon a time",
                tokenizer=tokenizer,
                max_new_tokens=256,
                top_k=50,
                temperature=0.8,
            )
            generated_text=generated_outputs["generated_text"]
            print_color(f"Generated text at step{step+1}:","cyan")
            print("Once upon a time",end="")
            print_color(f"{generated_text}\n","cyan")
        
        if train_config.wandb_logging and log_dict:
            wandb.log(log_dict,step=step+1)