import math
import torch
import tiktoken
import time

from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from einops import rearrange

@dataclass
class GPTConfig: 
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttentionOrig(nn.Module):

    def __init__(self, config):

        # without Falsh Attention

        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() 
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = rearrange(k, 'u v (w x) -> u w v x', w = self.n_head)
        q = rearrange(q, 'u v (w x) -> u w v x', w = self.n_head)
        v = rearrange(v, 'u v (w x) -> u w v x', w = self.n_head)

        sim = q@k.transpose(-1, -2)/math.sqrt(x.shape[-1])
        attn = sim.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim = -1)

        y = attn @ v
        y = rearrange(y, 'u w v x -> u v (w x)')

        y = self.c_proj(y)
        return y

class CasualSelfAttn(nn.Module):

    def __init__(self, config: GPTConfig):

        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.tensor):

        B, T, C = x.size() 
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y

class MLP(nn.Module): 
    def __init__(self, config: GPTConfig): 
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd) 
        self.act = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.tensor):
        
        x = self.c_fc(x) # B x block_size x 4*n_embd
        x = self.act(x) # B x block_size x 4*n_embd
        x = self.c_proj(x) # B x block_size x n_embd
        return x

class DecoderBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttn(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    '''
    embeddings -> Casual Self Attention -> MLP
    '''

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wpe = nn.Embedding(config.block_size, config.n_embd),
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # Weight Trying Scheme
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize the model
        self.apply(self._init_weights)
      
    def _init_weights(self, module):
        
        std = 0.02
        if isinstance(module, nn.Linear):

            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2*self.config.n_layer)**(-0.5)

            torch.nn.init.normal_(module.weight, std = std)

            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std = std)
    
    def configure_optmizers(self, lr = 6e-04, wd = 1e-01, betas = (0.9, 0.95), eps = 1e-08, device_type = device): 
        """

        Args:
          lr:
          wd:
          betas:
          eps:
          device_type:

        Returns:

        """

        # divide into two param groups
        # Assign weight decay to the ones with dim > 1

        decay_params = []
        non_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if param.dim() == 1: non_decay_params.append(param)
            else: decay_params.append(param)

        fused = True if device_type == 'cuda' else False
        optim = torch.optim.AdamW([
                                    {'params': decay_params, "weight_decay": wd}, 
                                    {'params': non_decay_params, "weight_decay": 0}, 
                                  ], lr = lr, betas = betas, eps = eps, fused = fused)
        return optim

    def forward(self, idx: torch.tensor, y: torch.tensor = None):
        """

        Args:
          idx:
          y:

        Returns:

        """
        
        # idx -> B, T
        # 1. Process Embeddings
        # 2. Iterative attention blocks
        # 3. Pass it through lm_head for final layer

        # assert idx.shape[-1] == self.config.block_size
        positions = torch.arange(0, idx.shape[-1], step = 1).to(idx.device)
        pos_embeddings = self.transformer.wpe(positions) # B, T
        tok_embeddings = self.transformer.wte(idx) # B, T, C

        x = pos_embeddings[None, :, :] + tok_embeddings

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))

        return  logits, loss