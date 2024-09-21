import numpy as np
import torch
import tiktoken

class DataLoaderLite:
    def __init__(self, B: int, T: int = 1024):
        self.enc = tiktoken.get_encoding('gpt2')
        with open('./input.txt', 'r') as f: data = f.read()
        tokens = self.enc.encode(data)
        self.tokens = torch.tensor(tokens, dtype = torch.long)
        self.B = B
        self.T = T
        self.current_position = 0
      
    def next_batch(self):
        if self.current_position + self.B*self.T > len(self.tokens): 
            self.current_position = 0

        offset = self.current_position + self.B*self.T + 1
        if offset > len(self.tokens): offset = len(self.tokens)

        buf = self.tokens[self.current_position:offset]
        x = buf[:-1].view(self.B, -1)
        y = buf[1:].view(self.B, -1)
        
        self.current_position += self.B*self.T
        if self.current_position > len(self.tokens): self.current_position = 0

        return x, y

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        