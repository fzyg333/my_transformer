import torch
import torch.nn.functional as F
from tokenizer_util import BPETokenizer
from transformer_util import TransformerLM

def top_k_sampling(
    logits:torch.Tensor,
    top_k:int,
):
    #按权重进行概率采样
    if top_k <=0:
        probs=F.softmax(logits,dim=-1)
        return torch.multinomial(probs,num_samples=1)

    top_k_logits,top_k_indices=torch.topk(logits,top_k,dim=-1)
    filtered_logits=torch.full_like(logits,float("-inf"))
    filtered_logits.scatter_(dim=-1,index=top_k_indices,src=top_k_logits)
    probs=F.softmax(filtered_logits,dim=-1)
    next_token=torch.multinomial(probs,num_samples=1)
    
    return next_token

def top_p_sampling(logits:torch.Tensor,top_p:float)->torch.Tensor:
    assert 0.0<top_p<=1.0,"top_p必须在0-1之间"
    
    sorted_logits,sorted_indices=torch.sort(logits,dim=-1,descending=True)
    sort_probs=F.softmax(sorted_logits,dim=-1)
    cum_probs=torch.cumsum(sort_probs,dim=-1)
    
    sorted_indices_to_remove=cum_probs>top_p
    sorted_indices_to_remove[...,1:]=sorted_indices_to_remove[...,:-1].clone()
    sorted_indices_to_remove[...,0]=False
    
    indices_to_remove=torch.zeros_like(logits,dtype=torch.bool)
    indices_to_remove.scatter_(dim=-1,index=sorted_indices,src=sorted_indices_to_remove)
    
    filtered_logits=logits.masked_fill(indices_to_remove,float("-inf"))
    
    probs=F.softmax(filtered_logits,dim=-1)
    return torch.multinomial(probs,num_samples=1)

@torch.no_grad()
def generate(
    model:TransformerLM,
    prompt:torch.Tensor | str,
    tokenizer:BPETokenizer,
    max_new_tokens:int=256,
    top_k:int=0,
    top_p:float=0.0,
    temperature:float=1.0,
)->dict:
    model.eval()
    if isinstance(prompt,str):
        out=tokenizer.encode(prompt)
        input_ids=out.ids if hasattr(out,"ids") else out
        input_ids=torch.tensor(input_ids,dtype=torch.long).unsqueeze(0)
    else:
        input_ids=prompt.unsqueeze(0)
    input_ids=input_ids.to(next(model.parameters()).device)#源代码似乎有误，不能直接model.device
    input_len=input_ids.shape[1]
    
    with torch.amp.autocast("cuda",enabled=False):
        for _ in range(max_new_tokens):
            logits,_=model(input_ids)
            next_token_logits=logits[:,-1,:].float()#获取最后一个token的得分
            
            assert temperature>0.0,"temperature must be positive"
            assert top_p==0.0 or top_k==0,"only one of top_p or top_k should be set"
            
            next_token_logits=next_token_logits/temperature
            
            if top_k>0:
                next_token_id=top_k_sampling(next_token_logits,top_k)
            elif top_p>0.0:
                next_token_id=top_p_sampling(next_token_logits,top_p)
            else:
                next_token_id=next_token_logits.argmax(dim=-1,keepdim=True)
            
            if next_token_id.item()==tokenizer.eos_token_id:
                break
            input_ids=torch.cat([input_ids,next_token_id],dim=-1)
    
    input_ids=input_ids.squeeze(0)#batch_size=1
    all_text=tokenizer.decode(input_ids.tolist())
    generated_ids=input_ids[input_len:]
    generated_text=tokenizer.decode(generated_ids.tolist())
    
    model.train()
    return {
        "all_text":all_text,
        "generated_text":generated_text,
        "generated_ids":generated_ids,
    }