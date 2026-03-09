from typing import BinaryIO
import os
from multiprocessing import Manager, Process, Queue
from collections import Counter,defaultdict
import regex as re
from queue import Empty
import heapq
import json
import numpy as np
from tqdm import tqdm, trange


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def spilt_by_special_tokens(text:str,special_tokens:list[str],include_special_tokens:bool=False)->list[str]:
    if not special_tokens:
        return [text]
    
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in special_tokens_sorted)
    
    if include_special_tokens:
        blocks=re.split(f"({pattern})",text)
    else:   
        blocks = re.split(pattern, text)
    
    return blocks

class BPETokenizer:
    def __init__(
        self,
        vocab:dict[int,bytes],
        merges:list[tuple[bytes,bytes]],
        special_tokens:list[str]|None=None
        ):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens if special_tokens else []
        self.special_tokens_bytes=[t.encode("utf-8") for t in self.special_tokens]
        self.special_set=set(self.special_tokens_bytes)
        
        self.vocab_inv={v:k for k,v in self.vocab.items()}
        rank:dict[tuple[int,int],int]={}
        merge_to_new_id:dict[tuple[int,int],int]={}
        
        for r,(a_bytes,b_bytes) in enumerate(self.merges):
            a_id=self.vocab_inv.get(a_bytes)
            b_id=self.vocab_inv.get(b_bytes)
            new_id=self.vocab_inv.get(a_bytes+b_bytes)
            if a_id is None or b_id is None or new_id is None:
                continue
            pair=(a_id,b_id)
            rank[pair]=r
            merge_to_new_id[pair]=new_id
        
        self.rank=rank
        self.merge_to_new_id=merge_to_new_id
        self.eos_token_id=self.vocab_inv.get(b"<|endoftext|>")

    def _pre_tokenize(self,text:str)->list[bytes]:
        parts=spilt_by_special_tokens(text,self.special_tokens,include_special_tokens=True)
        token_list:list[bytes]=[]
        
        for part in parts:
            if part=="":
                continue
            if part in self.special_tokens:
                token_list.append(part.encode("utf-8"))
            else:
                for token in re.findall(PAT,part):
                    token_list.append(token.encode("utf-8"))
            
        return token_list
    
    def encode(self,text:str)->list[int]:
        def merge_one_pretoken(ids:list[int])->list[int]:
            n=len(ids)
            if n<=1:
                return ids
            
            alive=[True]*n
            prev=[-1]*n
            nxt=[-1]*n
            for i in range(n):
                prev[i]=i-1
                nxt[i]=i+1 if i+1<n else -1
            
            heap:list[tuple[int,int]]=[]
            
            def push_if_vaild(i:int):
                cur_r=None
                j=nxt[i]
                if j==-1 or not alive[i] or not alive[j]:
                    cur_r=None
                else:
                    cur_r=self.rank.get((ids[i],ids[j]))
                
                if cur_r is not None:
                    heapq.heappush(heap,(cur_r,i))
            for i in range(n):
                push_if_vaild(i)
                
            while heap:
                r,i=heapq.heappop(heap)
                j=nxt[i]
                if j==-1 or not alive[i] or not alive[j]:
                    continue
                pair=(ids[i],ids[j])
                cur_r=self.rank.get(pair)
                if cur_r is None or cur_r!=r:
                    continue
                
                new_id=self.merge_to_new_id.get(pair)
                if new_id is None:
                    continue
                ids[i]=new_id
                
                alive[j]=False
                nj=nxt[j]
                nxt[i]=nj
                if nj!=-1:
                    prev[nj]=i
                
                pi=prev[i]
                if pi!=-1:
                    push_if_vaild(pi)
                push_if_vaild(i)

            out:list[int]=[]
            k=0
            while k!=-1:
                if alive[k]:
                    out.append(ids[k])
                k=nxt[k]
            
            return out
        
        byte_tokens=self._pre_tokenize(text)
        token_ids:list[int]=[]
        for token in byte_tokens:
            if token in self.special_set:
                token_ids.append(self.vocab_inv[token])
            else:
                ids=[self.vocab_inv[bytes([b])] for b in token]
                token_ids.extend(merge_one_pretoken(ids))
        
        return token_ids
    
    @classmethod
    def from_files(cls,vocab_path:str,merges_path:str,special_tokens:list[str])->"BPETokenizer":
        with open(vocab_path,encoding="utf-8") as f:
            vocab_data=json.load(f)
            vocab={int(i):bytes(v,"latin1") for v,i in vocab_data.items()}

        merges=[]
        with open(merges_path,encoding="utf-8") as f:
            x:int=0
            for line in f:
                parts=line.strip().split()
                x+=1
                if len(parts)==2:
                    merges.append((bytes(parts[0],"latin1"),bytes(parts[1],"latin1")))
        
        if isinstance(special_tokens,list):
            special_tokens_list=special_tokens
        else:
            special_tokens_list=[]
            
        return cls(vocab,merges,special_tokens_list)
    
    def decode(self,ids:list[int])->str:
        tokens=b"".join(self.vocab.get(i,b"\xef\xbf\xbd") for i in ids)
        return tokens.decode("utf-8",errors="replace")

def string_to_bytes(s: str,return_int:bool=False) ->  list[int]|list[bytes]:
    byte_array = s.encode("utf-8")
    if return_int:
        return list(map(int,byte_array))
    else:
        return  [bytes([b]) for b in byte_array]

def init_vocab(special_tokens: list[str] | None = None) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    index=256
    if special_tokens is not None:
        for token in special_tokens:
            vocab[index]=token.encode("utf-8")
            index+=1
    # print(vocab)
    return vocab

def find_boundaries(
    file:BinaryIO,
    block_num:int,
    split_token:bytes
)->list[int]:
    file.seek(0,os.SEEK_END)
    file_size=file.tell()
    file.seek(0)
    # print(file_size)
    block_size=file_size//block_num
    block_boundraies=[i*block_size for i in range(block_num+1)]
    block_boundraies[-1]=file_size
    per_read=4096
    for i in range(1,len(block_boundraies)-1):
        ini_pos=block_boundraies[i]
        file.seek(ini_pos)
        while True:
            read_data=file.read(per_read)
            if read_data==b"":
                block_boundraies[i]=file_size
                break
            
            find_at=read_data.find(split_token)
            if find_at!=-1:
                block_boundraies[i]=ini_pos+find_at
                break
            ini_pos+=per_read
            
    # file.seek(0)
    # for i in range(1,len(block_boundraies)):
    #     str=file.read(block_boundraies[i]-block_boundraies[i-1])
    #     print(str)
    # print(block_boundraies)
    
    return sorted(set(block_boundraies))

def pre_tokenizer_wordcounter(train_data_path,special_tokens,queue,start,end):
    with open(train_data_path,"rb") as f:
        f.seek(start)
        read_data=f.read(end-start).decode("utf-8",errors="ignore")
    word_counter=Counter()
    blocks:list[str]=[]
    if not special_tokens:
        blocks=[read_data]
    else:
        special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(t) for t in special_tokens_sorted)
        blocks = re.split(pattern, read_data)
    
    for block in blocks:
        for match in re.finditer(PAT,block):
            word=match.group(0)
            # print(word)
            word_encoded=tuple(string_to_bytes(word,return_int=True))
            word_counter[word_encoded]+=1
    print("运行完毕")
    # print(word_counter)
    queue.put(word_counter)
    

class HeapItem:
    def __init__(self, freq: int, pair_bytes: tuple[int,int]):
        self.freq = freq
        self.pair_bytes = pair_bytes

    def __lt__(self, other: "HeapItem") -> bool:
        if self.freq != other.freq:
            return self.freq < other.freq
        else:
            return self.pair_bytes > other.pair_bytes  


def bulid_pair_heap(pairs_counter:Counter):
    heap=[]
    for (a,b),f in pairs_counter.items():
        if f>0:
            item=HeapItem(-f,(a,b))
            heapq.heappush(heap,item)
    
    # heapq.heapify(heap)
    return heap


def pop_mostfrequent_pair(pair_heap:list[HeapItem],pairs_counter:Counter):
    while pair_heap:
        item=pair_heap[0]
        freq=item.freq
        pair_bytes=item.pair_bytes
        real_freq=pairs_counter.get(pair_bytes,0)
        if real_freq<=0 or -freq!=real_freq:
            heapq.heappop(pair_heap)
            continue
        return pair_bytes
    
    raise ValueError("heap is None")


def get_new_word(word:tuple[int,...],target_pair:tuple[int,int],new_id:int)-> tuple[int,...]:
    a,b=target_pair
    new_word=[]
    i=0
    while i<len(word):
        if i+1<len(word) and word[i]==a and word[i+1]==b:
            new_word.append(new_id)
            i+=2
        else:
            new_word.append(word[i])
            i+=1
    return tuple(new_word)

def merge_pairs(
    word_counter:Counter,
    pair_counter:Counter,
    pair_to_words:dict[tuple[int,int],set[tuple[int,...]]],
    target_pair:tuple[int,int],
    new_id:int,
    pair_heap,
):
    new_word_counter:Counter=Counter(word_counter)
    update_pair_counter:Counter=pair_counter.copy()
    changed_pairs:set[tuple[int,int]]=set()
    affected_words=list(pair_to_words.get(target_pair,set()))
    for word in affected_words:
        word_num=word_counter.get(word,0)
        if word_num<=0 or len(word)<2:
            continue
        
        # ???
        new_word_counter[word]-=word_num
        if new_word_counter[word]<=0:
            del new_word_counter[word]
        
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            update_pair_counter[pair]-=word_num
            changed_pairs.add(pair)
            
            s=pair_to_words.get(pair)
            if s is not None:
                s.discard(word)
                if not s:
                    del pair_to_words[pair]
        
        new_word=get_new_word(word,target_pair,new_id)
        new_word_counter[new_word]+=word_num
        
        if len(new_word)>=2:
            for i in range(len(new_word)-1):
                pair=(new_word[i],new_word[i+1])
                update_pair_counter[pair]+=word_num
                changed_pairs.add(pair)
                pair_to_words.setdefault(pair,set()).add(new_word)
                
    if pair_heap is not None:
        for pair in changed_pairs:
            f=update_pair_counter.get(pair,0)
            if f>0:
                heapq.heappush(pair_heap,HeapItem(-f,pair))        
    
    return dict(new_word_counter),update_pair_counter,pair_to_words,pair_heap   

def update_vocab(vocab:dict[int,bytes],most_frequent_pair:tuple[int,int]):
    new_id=len(vocab)
    vocab[new_id]=vocab[most_frequent_pair[0]]+vocab[most_frequent_pair[1]]
    return new_id

def save_vocab_and_merges(vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],save_path:str):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    vocab_filepath=os.path.join(save_path,"vocab.json")
    merge_filepath=os.path.join(save_path,"merges.txt")
    vocab_inv={v.decode("latin1"):k for k,v in vocab.items()}
    with open(vocab_filepath,"w",encoding="utf-8") as vf:
        json.dump(vocab_inv,vf,ensure_ascii=False,indent=2)
    with open(merge_filepath,"w",encoding="utf-8") as mf:
        for a,b in merges:
            mf.write(f"{a.decode("latin1")} {b.decode("latin1")}\n")


def train_bpe(
    train_data_path: str,
    save_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    sign: bool = False,
) -> dict[int, bytes]:
    ite_num = vocab_size - 256 - (len(special_tokens) if special_tokens else 0)
    vocab: dict[int, bytes] = init_vocab(special_tokens)
    merges:list[tuple[bytes,bytes]]=[]
    with open(train_data_path,"rb") as f:
        block_boundaries=find_boundaries(f,4,b"\n")
    manager=Manager()
    queue=manager.Queue()
    process:list[Process]=[]
    for start,end in zip(block_boundaries[:-1],block_boundaries[1:]):
        p=Process(
            target=pre_tokenizer_wordcounter,
            args=(train_data_path,special_tokens,queue,start,end)
        )
        process.append(p)
        p.start()
        
    for p in process:
        p.join()
        
    word_counter = Counter()
    for _ in range(len(process)):
        try:
            partial_counter = queue.get(timeout=10)
            word_counter.update(partial_counter)
        except Empty:
            continue
    
    pairs_counter=Counter()
    pair_to_words:dict[tuple[int,int],set[tuple[int,...]]]=defaultdict(set)
    for word in word_counter:
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            pair_to_words[pair].add(word)
            pairs_counter[pair] += word_counter[word]
    # print(pairs_counter)
    # print("1111")
    pair_heap:list[HeapItem]=bulid_pair_heap(pairs_counter)
    # print("堆建立成功")
    # sorted_elements = heapq.nsmallest(len(pair_heap), pair_heap)
    # for item in sorted_elements:
    #     print(item.freq)
    #     print(vocab[item.pair_bytes[0]],end='')
    #     print(vocab[item.pair_bytes[1]])
    for i in range(ite_num):
        most_frequent_pair=pop_mostfrequent_pair(pair_heap,pairs_counter)
        new_id=update_vocab(vocab,most_frequent_pair)
        word_counter,pairs_counter,pair_to_words,pair_heap=merge_pairs(
            word_counter,pairs_counter,pair_to_words,most_frequent_pair,new_id,pair_heap
        )
        merges.append((vocab[most_frequent_pair[0]],vocab[most_frequent_pair[1]]))
    
    save_vocab_and_merges(vocab,merges,save_path)
    
    return vocab,merges


def load_tokenizer_from_dir(save_path:str,special_tokens:list[str])->BPETokenizer:
    vocab_path=os.path.join(save_path,"vocab.json")
    merges_path=os.path.join(save_path,"merges.txt")
    tokenizer=BPETokenizer.from_files(vocab_path,merges_path,special_tokens)
    return tokenizer

def encode_file_to_bin(tokenizer:BPETokenizer,data_path,out_bin_path,dtype=np.uint16):
    total_bytes=os.path.getsize(data_path)
    
    with open(data_path,encoding="utf-8") as f_in,open(out_bin_path,"wb") as f_out:
        process_bar=tqdm(total=total_bytes,desc="转换为二进制",unit="B",unit_scale=True) 
        
        for line in f_in:
            token_ids=tokenizer.encode(line)
            arr=np.array(token_ids,dtype=dtype)
            arr.tofile(f_out)
            process_bar.update(len(line.encode("utf-8")))
            
def decode_bin_to_file(tokenizer:BPETokenizer,bin_path,out_txt_path,dtype=np.uint16):
    data_int=np.fromfile(bin_path,dtype=np.uint16)
    data_bytes=b''
    for token_int in data_int:
        data_bytes+=tokenizer.vocab[token_int]
    decode_data=data_bytes.decode("utf-8")
    with open(out_txt_path,'w',encoding="utf-8") as f:
        f.write(decode_data)

def load_tokenizer_from_dir(dir_path:str)->BPETokenizer:
    vocab_path=os.path.join(dir_path,"vocab.json")
    merges_path=os.path.join(dir_path,"merges.txt")
    # special_tokens_path=os.path.join(dir_path,"special_tokens.txt")
    tokenizer=BPETokenizer.from_files(vocab_path,merges_path,["<|endoftext|>"])
    return tokenizer