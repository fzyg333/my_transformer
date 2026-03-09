import os
import numpy as np
from tqdm import tqdm
from tokenizer_util import train_bpe,load_tokenizer_from_dir,encode_file_to_bin,decode_bin_to_file

SETTING = {
    "train_data_path": "data/TinyStoriesV2-GPT4-train.txt",
    "dev_data_path": "data/TinyStoriesV2-GPT4-valid.txt",
    "vocab_size": 10000,
    "special_tokens": ["<|endoftext|>"],
    "save_path": "./result",
}

if __name__ == "__main__":
    setting = SETTING
    if not os.path.exists(setting["save_path"]):
        os.makedirs(setting["save_path"])
        train_bpe(
            train_data_path=setting["train_data_path"],
            save_path=setting["save_path"],
            vocab_size=setting["vocab_size"],
            special_tokens=setting["special_tokens"],
            sign=True,
            )
        print("BPE分词器运行成功")
    else:
        pass
        # train_bpe(
        #     train_data_path=setting["train_data_path"],
        #     save_path=setting["save_path"],
        #     vocab_size=setting["vocab_size"],
        #     special_tokens=setting["special_tokens"],
        #     sign=True,
        #     )
        # print("BPE分词器运行成功")
        
    
    tokenizer=load_tokenizer_from_dir(setting["save_path"])
    
    out_bin_path=os.path.join(setting["save_path"],"train.bin")
    encode_file_to_bin(tokenizer,setting["train_data_path"],out_bin_path,dtype=np.uint16)
    print("训练数据编码成功")
    eval_bin_path=os.path.join(setting["save_path"],"eval.bin")
    encode_file_to_bin(tokenizer,setting["dev_data_path"],eval_bin_path,dtype=np.uint16)
    print("评估数据编码成功")
    # decode_bin_to_file(tokenizer,out_bin_path,os.path.join(setting["save_path"],"decode_train.txt"))
    # decode_bin_to_file(tokenizer,eval_bin_path,os.path.join(setting["save_path"],"decode_eval.txt"))