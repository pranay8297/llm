import math
import torch

from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from einops import rearrange, repeat

@dataclass
class ESMConfig(): 
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    hidden_size: int = 4096 # 4 * block_size
    dropout: float = 0.0

class ESMIntermediateLayer(nn.Module):
    def __init__(self, nin, nout, dropout = 0.0):
        super().__init__()
        self.dense = nn.Linear(nin, nout)
        self.act = nn.GELU(approximate = 'tanh')

    def forward(self, x):
        return self.act(self.dense(x))

class ESMOutLayer(nn.Module):
    def __init__(self, nin, nout, dropout = 0.0):
        super().__init__()
        self.dense = nn.Linear(nin, nout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_scores):
        x = self.dense(x)
        x = self.dropout(x)
        x = x + attn_scores
        return x

class ESMSelfAttn(nn.Module): # Verified
    
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        attention_head_size = config.n_embd//config.n_head
        # Add a rotary embeddings here
        self.rotary_embeddings = RotaryEmbedding(dim = attention_head_size)
     
    def forward(self, x, attention_mask): 

        # x -> (b, s, e) -> (b s, h, e/h)
        k, q, v = self.key(x), self.query(x), self.value(x)

        k = rearrange(k, 'b s (h e) -> b h s e', h = self.n_head)
        q = rearrange(q, 'b s (h e) -> b h s e', h = self.n_head)
        v = rearrange(v, 'b s (h e) -> b h s e', h = self.n_head)

        # Add rotary embeddings here for k and q tensors
        q, k = self.rotary_embeddings(q, k)

        # Attention claculation - # TODO: make is_casual true in case of finetuning - Very important
        y = F.scaled_dot_product_attention(q, k, v, attn_mask = attention_mask, is_causal=False) # flash attention 
        y = rearrange(y, 'b h s e -> b s (h e)', h = self.n_head)
        return y

class ESMAttn(nn.Module): # Verified

    def __init__(self, config):
        super().__init__() # No activation function at this level
        self.self = ESMSelfAttn(config)
        self.output = ESMOutLayer(config.n_embd, config.n_embd, dropout = getattr(config, 'dropout', 0.))
        self.LayerNorm = nn.LayerNorm(config.n_embd)
    
    def forward(self, x, attention_mask):
        inter_x = self.LayerNorm(x)
        attn = self.self(inter_x, attention_mask)
        out = self.output(attn, x)
        return out

class ESMLayers(nn.Module): # Both Init and Forward Verified - Done and Dusted

    def __init__(self, config):
        super().__init__()
        self.attention = ESMAttn(config) 
        self.intermediate = ESMIntermediateLayer(config.n_embd, config.hidden_size) # 
        self.output = ESMOutLayer(config.hidden_size, config.n_embd) #
        self.LayerNorm = nn.LayerNorm(config.n_embd)
    
    def forward(self, x, attention_mask):
        attention_op = self.attention(x, attention_mask)
        attention_op_ln = self.LayerNorm(attention_op)
        inter = self.intermediate(attention_op_ln)
        out = self.output(inter, attention_op)
        return out

class ESMEncoder(nn.Module):

    def __init__(self, config):
        super().__init__() 
        
        # No activation functions here as well

        self.layer = nn.ModuleList([ESMLayers(config) for _ in range(config.n_layer)])
        self.emb_layer_norm_after = nn.LayerNorm(config.n_embd)
    
    def forward(self, x, attention_mask = None): 

        for layer in self.layer:
            x = layer(x, attention_mask)

        return self.emb_layer_norm_after(x)

class ESM(nn.Module):
    
    def __init__(self, config):
        
        super().__init__()
        self.config = config
        self.esm = nn.ModuleDict(dict(
            embeddings = nn.ModuleDict(dict( # No activation needed - for both the embeddings - done - forward done
                word_embeddings = nn.Embedding(config.vocab_size, config.n_embd),
                position_embeddings = nn.Embedding(config.block_size, config.n_embd)
            )),
            encoder = ESMEncoder(config), # Done, forward - here
            final_layer = nn.Linear(config.n_embd, config.vocab_size)
        ))
        self.esm.final_layer.weight = self.esm.embeddings.word_embeddings.weight
    
    @classmethod
    def get_pretrained_config(cls, model_type = 'esm2_t33_650M_UR50D'):

        '''
        name                n_layers    n_params  
        esm2_t48_15B_UR50D	48	        15B
        esm2_t36_3B_UR50D	36	        3B
        esm2_t33_650M_UR50D	33	        650M
        esm2_t30_150M_UR50D	30	        150M
        esm2_t12_35M_UR50D	12	        35M
        esm2_t6_8M_UR50D
        '''

        assert model_type in {'esm2_t36_3B_UR50D', 'esm2_t33_650M_UR50D', 'esm2_t30_150M_UR50D'}

        config_args = {
            'esm2_t36_3B_UR50D': dict(n_layer=36, n_head = 40, n_embd=2560, hidden_size=10240), # 3B params
            'esm2_t33_650M_UR50D': dict(n_layer=33, n_head = 20, n_embd=1280, hidden_size=5120), # 650M params
            'esm2_t30_150M_UR50D': dict(n_layer=30, n_head = 20, n_embd=640, hidden_size=2560), # 150M params
        }[model_type]

        config_args['vocab_size'] = 33 # always 33 for ESM Models
        config_args['block_size'] = 1026 # Always constant for ESM Models
        config = ESMConfig(**config_args)
        return config

    
    @classmethod
    def load_pretrained(cls, model_type = 'esm2_t33_650M_UR50D'):

        config = cls.get_pretrained_config(model_type)
        print("loading weights from pretrained gpt: %s" % model_type)

        # create a from-scratch initialized minGPT model
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        from transformers import AutoModelForSequenceClassification

        model_checkpoint = f"facebook/{model_type}"
        num_labels = 33
        model_hf = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        sd_keys_hf = [k for k in sd_keys_hf if 'inv_freq' not in k]
        sd_keys_hf = [k for k in sd_keys_hf if 'classifier' not in k]

        ignore_keys = ['esm.contact_head.regression.weight', 'esm.contact_head.regression.bias']
        for k in sd_keys_hf:
            
            if k in ignore_keys: continue

            # vanilla copy over the other parameters
            try: assert sd_hf[k].shape == sd[k].shape
            except Exception as e: print(f"Mismatch in the shape of tensor while loading weights - Key: {k}, expected shape: {sd_hf[k].shape}, actual shape: {sd[k].shape}")
            
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        # Set the final layers bias as 0 so that it does not affect weight tying scheme
        with torch.no_grad():
            model.esm.final_layer.bias.zero_()
        
        # Freeze the model
        for param in model.parameters():
            param.requires_grad = False

        return model

    def get_embs(self, x, attention_mask = None):
        token_embs = self.esm.embeddings.word_embeddings(x)
        # Not required as we are use rotary embeddings - Hence we do not require absolute position embeddings
        # position_embs = self.esm.embeddings.position_embeddings(torch.arange(0, x.shape[1], 1, dtype = torch.long)) 

        if attention_mask is not None: 
            token_embs = (token_embs * attention_mask.unsqueeze(-1)).to(token_embs.dtype)

        return token_embs 

    def get_extended_attn_mask(self, attention_mask, input_shape):
        
        if attention_mask == None: return None
        b, s = attention_mask.shape
        # Make the attention mask braodcastable for [batch_size, n_heads, seq_len, seq_len]
        attention_mask = attention_mask[:, None, None, :]      

        # Now make sure that it has negetive infinity for all the padded tokens and 
        # 0 for all attention tokens as we add this mask to attention scores
        attn_mask = attention_mask.to(torch.float32)
        attn_mask = (1 - attn_mask) * (torch.finfo(torch.float32).min)
        attn_mask = attn_mask.expand(b, 1, s, s)
        return attn_mask

    def forward(self, x, y = None, attention_mask = None, output_encoder = True):

        # Calculate Embeddings
        x = self.get_embs(x, attention_mask) # Verified

        # compute attention_mask for attention scores
        attention_mask = self.get_extended_attn_mask(attention_mask, x.shape)

        #Do the forward pass
        x = self.esm.encoder(x, attention_mask = attention_mask)
        if not output_encoder:
            return self.esm.final_layer(x)
        else:
            return self.esm.final_layer(x), x

model = ESM.load_pretrained("esm2_t30_150M_UR50D") # load the pretrained frozen model
print('Models created')
print(model)

# Next: TODO - Load the pretrained weights to this model - Done
# Add required Activation functions and all... make sure its the same forward as of origina esm model's
# Implement Rotary Embeddings - Done
# Do a forward pass - Partially done - verification process - Done

# Then work on Embeddings - Add new tokens - Keep the existing tokens - Turn on requires grad
# Get the training data
# Set up Lora for the model
# Finetune - Hope for the best - Snowflake