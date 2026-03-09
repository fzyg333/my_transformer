from generator import generate
from transformer_util import TransformerLM,ModelConfig,TrainConfig
import torch
from tokenizer_util import BPETokenizer,load_tokenizer_from_dir

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

#加载模型
model_config_json="checkpoints/my_transformer/model_config.json"
train_config_json="checkpoints/my_transformer/train_config.json"
model_config=ModelConfig.from_json(model_config_json)
train_config=TrainConfig.from_json(train_config_json)
model=TransformerLM(model_config)
model=model.to(device=model_config.device)
checkpoint=torch.load("checkpoints/my_transformer/best_model_step_1000.pt",map_location=model_config.device)#注意checkpoint路径需要修改
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

tokenizer=load_tokenizer_from_dir(train_config.dataset_dir)

with torch.no_grad():
    prompt=input("请输入故事开头:")
    if isinstance(prompt,str):
        generated_outputs=generate(
                model=model,
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=256,
                top_k=50,
                temperature=0.8,
            )
        generated_text=generated_outputs["generated_text"]
        print(prompt,end="")
        print_color(f"{generated_text}\n","cyan")
    else:
        print("输入prompt类型错误")