import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """
    A causal self-attention module.
    """
    def __init__(self, n_embd, n_head, block_size, dropout, bias):
        super().__init__()

        # This is necessary because head_size = n_embd // n_head
        assert n_embd % n_head == 0

        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout

        # Key, query, value projections for all heads in a batched linear layer;
        # i.e. one big linear layer that computes Q, K, V for all heads at once.
        # Input: (B, T, n_embd)
        # Output: (B, T, 3*n_embd), which can be split into Q, K, V
        # - Each Q, K, V contains the Q, K, V for all heads.
        # - Shape of Q, K, V is (B, T, n_embd), which is equivalent to (B, T, n_head, head_size).
        #   In forward(), we will transpose to get (B, n_head, T, head_size).
        self.attention_layer = nn.Linear(n_embd, 3 * n_embd, bias=bias)

        # Output projection: basically "mixing" the heads' outputs together.
        # Input: (B, T, n_embd), i.e. (B, T, n_head * head_size)
        # Output: (B, T, n_embd)
        self.projection = nn.Linear(n_embd, n_embd, bias)

        # Regularization: dropout to prevent overfitting
        self.attention_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: ensures that attention is only applied to the left ("previous")
        # tokens in the input sequence.
        tril = torch.tril(torch.ones(block_size, block_size)) # (T, T)
        # Introduce batch and head dimensions to allow broadcasting
        tril = tril.view(1, 1, block_size, block_size)        # (1, 1, T, T)
        # Tell pytorch to not compute gradients for it
        self.register_buffer("causal_mask", tril)
    
    def forward(self, input):
        # B = batch size
        # T = time = sequence length
        # C = channel = embedding dimensionality (n_embd)
        B, T, C = input.size()

        # Get attention layer output and split into Q, K, V
        # Input: (B, T, n_embd)
        # Output: (B, T, 3 * n_embd)
        # Split into Q, K, V: (B, T, n_embd), (B, T, n_embd), (B, T, n_embd)
        q, k, v = self.attention_layer(input).split(self.n_embd, dim=2)

        # Reshape Q, K, V into (B, T, n_head, head_size);
        # then transpose into (B, n_head, T, head_size)
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, n_head, T, head_size)

        # Implement attention:

        # (1) Compute attention scores ("affinities")
        # (B, n_head, T, head_size) @ (B, n_head, head_size, T) => (B, n_head, T, T)
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        weights = weights.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1) # (B, n_head, T, T)
        
        # Dropout layer
        weights = self.attention_dropout(weights)

        # Output: attention_weights @ value
        # (B, n_head, T, T) @ (B, n_head, T, head_size) => (B, n_head, T, head_size)
        output = weights @ v

        # Reassemble all head outputs side by side
        # note: .contiguous() is required to call .view()
        output = output.transpose(1, 2).contiguous() # (B, T, n_head, head_size)
        output = output.view(B, T, C)                # (B, T, C) = (B, T, n_embd)

        # Output projection: basically "mixing" the heads' outputs together
        output = self.projection(output)

        # Residual dropout:
        # this dropout is applied to the output before residual connection (in Block)
        output = self.resid_dropout(output)
        return output

class LayerNorm(nn.Module):
    """
    In older version of PyTorch, nn.LayerNorm doesn't support bias=False.
    So here, we write a customized LayerNorm with an optional bias.
    """
    def __init__(self, n_dim, bias = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    """
    This is the Feed-Forward Network (FFN) module.
    It consists of 3 layers:
    - Linear layer (expansion)
    - GELU activation function
    - Linear layer (projection)
    """
    def __init__(self, n_embd, dropout, bias = False):
        super().__init__()

        # Layer 1: linear layer (expansion)
        # - expands dimensions from n_embd to 4 * n_embd
        #   (B, T, n_embd) -> (B, T, 4 * n_embd)
        # - "cfc" = "causal fully connected" (naming from GPT-2)
        # - creates a wider hidden layer for more expressive computation
        self.cfc = nn.Linear(n_embd, 4 * n_embd, bias=bias)

        # Layer 2: non-linear activation (Gaussian Error Linear Unit)
        # - smoother / performs better than ReLU
        self.gelu = nn.GELU()

        # Layer 3: linear layer (projection)
        # - projects dimensions back to n_embd
        #   (B, T, 4 * n_embd) -> (B, T, n_embd)
        # - returns to original dimension to allow for residual connection
        self.projection = nn.Linear(4 * n_embd, n_embd, bias=bias)

        # Dropout layer: prevents overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.cfc(x)
        x = self.gelu(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    A transformer block that consists of:
    - Layer Normalization
    - Causal Self-Attention
    - Layer Normalization
    - Feed-Forward Network (MLP)
    """
    def __init__(self, n_embd, n_head, block_size, dropout, bias = False):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x):
        # Apply residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x