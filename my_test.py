# from typing import BinaryIO
# import regex as re
# import sys
import torch
# import wandb
# import dotenv
# import os
# import einops
# import torch.nn as nn
# from dataclasses import dataclass
# from generator import top_p_sampling

# def print_color(content: str, color: str = "green"):
#     colors = {
#         "black": "\033[30m",
#         "red": "\033[31m",
#         "green": "\033[32m",
#         "yellow": "\033[33m",
#         "blue": "\033[34m",
#         "magenta": "\033[35m",
#         "cyan": "\033[36m",
#         "white": "\033[37m",
#     }
#     reset = "\033[0m"
#     print(f"{colors.get(color, '')}{content}{reset}")

# print_color("i am xiaopangwa","cyan")
# x=torch.tensor([1,2])
# x=x.to(device="cuda")
# print(x.device)
x=torch.ones(5)
print(x)
print(x.size)