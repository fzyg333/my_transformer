import os
os.environ["PYTHONUTF8"] = "1"
from transformer_util import ModelConfig,TrainConfig
import dotenv
from transformer_util import seed_everything,TransformerLM,print_color
from adamw import AdamW
from train_util import train


def main(
    train_config_json:str | None=None,
    model_config_json:str | None=None,
):
    train_config=TrainConfig.from_json(train_config_json) if train_config_json else TrainConfig()
    model_config=ModelConfig.from_json(model_config_json) if model_config_json else ModelConfig()
    
    out_dir=os.path.join(
        train_config.save_checkpoint_dir,
        train_config.model_name
    )
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    model_config.to_json(os.path.join(out_dir, "model_config.json"))
    train_config.to_json(os.path.join(out_dir, "train_config.json"))
    #启动wandb
    dotenv.load_dotenv("wandb.env")
    wandb_api=os.getenv("WANDB_API_KEY")
    if train_config.wandb_logging and wandb_api is None:
        raise ValueError("wandb密钥获取失败")
    if train_config.wandb_logging:
        import wandb
        wandb.login(key=wandb_api)
        wandb.init(
            project="my_transformer",
            name=train_config.model_name+f"_batch-{train_config.batch_size}_steps-{train_config.num_steps}",
            config={
                "model_config":model_config.to_dict(),
                "train_config":train_config.to_dict()
            }
        )
    
    seed_everything(train_config.seed)
    
    #模型架构
    model=TransformerLM(model_config)
    model=model.to(train_config.device)
    model.train()
    
    #优化器
    optimizer=AdamW(
        model.parameters(),
        lr=train_config.min_lr,
        betas=train_config.betas,
        weight_decay=train_config.weight_decay
    )
    
    #开始训练
    print_color("开始训练>>>","blue")
    print_color(f"总共训练轮次:{train_config.num_steps}","blue")
    train(model=model,optimizer=optimizer,train_config=train_config)
    print("训练完成")
    
    if train_config.wandb_logging:
        wandb.finish()

if __name__=="__main__":
    main("checkpoints/my_transformer/train_config.json","checkpoints/my_transformer/model_config.json")