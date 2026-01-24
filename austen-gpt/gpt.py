from torch.nn.modules.module import Module


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
from block import Block, LayerNorm

class GPT(nn.Module):
    """
    A GPT model that consists of:
    - Embedding layers (token embeddings and position embeddings)
    - Transformer blocks (multiple blocks of attention and MLP)
    - Layer normalization
    - Linear layer (output layer)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Transformer components
        self.transformer = nn.ModuleDict(dict[str, Module](
            # Word token embedding: maps token IDs to embeddings;
            # each of the `vocab_size` tokens gets mapped to n_embd dimensions
            wte = nn.Embedding(config.vocab_size, config.n_embd),

            # Word position embedding: maps token positions to embeddings
            # each of the `block_size` positions gets mapped to n_embd dimensions
            wpe = nn.Embedding(config.block_size, config.n_embd),

            # Dropout layer: applied after adding token + position embeddings
            drop = nn.Dropout(config.dropout),

            # Hidden layers: the transformer blocks;
            # each block has attention + FFN
            hidden = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            # LayerNorm final: applied after all transformer blocks
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Language-modeling head: maps embeddings (of dimension `n_embd`) to
        # logits (dimension `vocab_size`) to predict the next token
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying:
        # Shares weights between input embedding (wte) and output projection (lm_head)
        # - wte: maps token ID -> embedding
        # - lm_head: maps embedding -> token logits
        # Using same weights: "if tokens have similar embeddings, they should have similar
        #              probabilities when predicting the next token (i.e. similar logits)"
        # Benefits: fewer parameters
        self.transformer.wte.weight = self.lm_head.weight

        # Weight initialization
        # - applies _init_weights function to all parameters
        # - typically initializes with small random values (e.g., normal distribution)
        self.apply(self._init_weights)

        # Apply special scaled initialization to the residual projections, per GPT-2 paper
        # - finds all `projection` layers (in attention and FFN)
        # - initializes them with smaller standard deviation
        # - formula: std = 0.02 / sqrt(2 * n_layer)
        # Why?
        # - deeper networks (more layers) need smaller initial weights
        # - prevents values from exploding as they pass through many residual connections
        # - helps training stability
        for pn, p in self.named_parameters():
            if pn.endswith('projection.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Total number of parameters
        print("Total number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, exclude_embeddings=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (exclude_embeddings=True), the position embeddings get removed.
        We should also remove the token embeddings, but due to weight-sharing, these params are
        # also used as weights in the final layer (lm_head), so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        Initializes weights to random values from normal distribution.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # Linear layer may have a bias b: output = Wx + b
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        """
        Forward pass of the GPT model.
        input_ids: input token ids, (batch_size, sequence_length)
        targets: target token ids, (batch_size, sequence_length)
        """
        device = input_ids.device
        b, t = input_ids.size()
        
        # Check that sequence length is not greater than max context
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t} because block size is only {self.config.block_size}"
        
        # Position indices: [0, 1, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device) # (t)

        # Forward the GPT model:
        # 1. Add token and position embeddings (with dropout)
        tok_emb = self.transformer.wte(input_ids)    # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)          # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb) # (b, t, n_embd)

        # 2. Go through hidden layers (transformer blocks)
        for block in self.transformer.hidden:
            x = block(x)                            # (b, t, n_embd)

        # 3. Apply final LayerNorm
        x = self.transformer.ln_f(x)                # (b, t, n_embd)

        if targets is not None:
            # Training mode: we are given desired targets
            # - compute logits for all positions (since we need loss for every token)
            logits = self.lm_head(x)                # (b, t, vocab_size)

            # Flatten logits and targets to compute cross-entropy loss
            # - logits.view(-1, vocab_size) => flatten to (b*t, vocab_size)
            # - targets.view(-1) => flatten to (b*t)
            # - ignore_index=-1 => don't compute loss for padding tokens (i.e. target = -1)
            #   padding tokens are used for padding the sequence to the same length as the input
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode: no targets
            # - optimization: only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
                                                 # (b, 1, vocab_size)
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Create an optimizer with different weight decay settings for different parameter types.
        """

        # Get all parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # Filter out parameters that do not require gradients
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Split into 2 optimization groups:
        # - decay_params: any parameters with dimensions >= 2 will be weight-decayed
        #   e.g. weight matrices in linear layers, embedding tables
        # - nodecay_params: any parameters with dimensions < 2 will not be weight-decayed
        #   e.g. biases, layer normalization weights and biases
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Print statistics
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if it is available
        # Fused AdamW:
        # - faster implementation on CUDA
        # - combines multiple operations into one kernel
        # - only available on GPU with newer PyTorch
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW optimizer: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates tokens autoregressively, one at a time, and feeding output back into the model.
        - `idx`: starting sequence, shape (batch_size, seq_len)
        - `max_new_tokens`: how many tokens to generate
        - `temperature`: controls randomness (higher = more random)
        - `top_k`: limits choices to top k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Crop the context if it's growing longer than block_size
            idx_context = idx
            if idx.size(1) > self.config.block_size:
                idx_context = idx[:, -self.config.block_size:]

            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_context) # (b, t, vocab_size)

            # Get the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature

            # Crop the logits to only the top k options
            if top_k is not None:
                top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Replace with -inf if the logit is less than the k-th highest logit
                logits[logits < top_k_logits[:, [-1]]] = -float('Inf')

            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx