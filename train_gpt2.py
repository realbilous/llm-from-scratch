from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention module that implements the masked multi-head attention mechanism.
    In causal self-attention, each token can only attend to itself and previous tokens.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Combined projection for Q, K, V - projects from n_embd to 3 * n_embd (for Q, K, V)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Create causal mask to ensure tokens can only attend to previous tokens
        # Shape: (1, 1, block_size, block_size)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size)) 
    
    def forward(self, x):
        # x shape: (B, T, C) where:
        # B = batch size
        # T = sequence length
        # C = embedding dimension (n_embd)
        B, T, C = x.size()

        # Project input to Q, K, V
        # qkv shape: (B, T, 3 * C)
        qkv = self.c_attn(x)
        # Split into Q, K, V
        # Each has shape: (B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape Q, K, V for multi-head attention
        # Shape becomes: (B, nh, T, hs) where:
        # nh = number of heads
        # hs = head size (C // n_head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Compute attention scores
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Apply causal mask: sets attention scores to -inf where mask is 0
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # Apply softmax to get attention weights
        att = F.softmax(att, dim=-1)
        
        # Apply attention to values
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        # Reshape back to (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        # Shape remains (B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """
    Multi-layer perceptron with GELU activation.
    Expands embedding dimension by 4x then projects back.
    """
    def __init__(self, config):
        super().__init__()
        # Input projection: n_embd -> 4 * n_embd
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        # Output projection: 4 * n_embd -> n_embd
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # x shape: (B, T, C)
        x = self.c_fc(x)    # -> (B, T, 4C)
        x = self.gelu(x)    # -> (B, T, 4C)
        x = self.c_proj(x)  # -> (B, T, C)
        return x

class Block(nn.Module):
    """
    Transformer block: LayerNorm -> Attention -> LayerNorm -> MLP
    Each with residual connections
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # x shape: (B, T, C)
        x = x + self.attn(self.ln_1(x))  # Residual connection
        x = x + self.mlp(self.ln_2(x))   # Residual connection
        return x

@dataclass
class GPTConfig:
    """Configuration class for GPT model hyperparameters"""
    block_size: int = 1024    # Maximum sequence length
    vocab_size: int = 50257   # Size of vocabulary
    n_layer: int = 12        # Number of transformer blocks
    n_head: int = 12         # Number of attention heads
    n_embd: int = 768        # Embedding dimension

class GPT(nn.Module):
    """
    GPT Language Model with token and position embeddings,
    followed by a stack of transformer blocks
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # Position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),  # Final layer norm
        ))

        # Project to vocabulary size for next token prediction
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        # idx shape: (B, T) - Input token indices
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
    
        # Generate position indices and get embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emd = self.transformer.wpe(pos)  # (T, C)
        tok_emd = self.transformer.wte(idx)  # (B, T, C)
        x = tok_emd + pos_emd                # (B, T, C) - Implicitly broadcasting position embeddings
        
        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x)                             # (B, T, C)
        
        x = self.transformer.ln_f(x)                 # (B, T, C)
        # Project to vocabulary size
        logits = self.lm_head(x)                    # (B, T, vocab_size)
        return logits
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# ------------------------------------------------------------------------------------------------

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
print("Model loaded")
model.eval()
model.to('cuda')

import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to('cuda')

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :] 
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, num_samples=1)
        xcol = torch.gather(topk_indices, dim=-1, index=ix)
        x = torch.cat((x, xcol), dim=1)


for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print('-' * 40)
    # print(tokens)
    print(decoded)


