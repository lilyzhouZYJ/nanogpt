# Training Configuration

This document describes all configuration parameters available in `train_config.yaml`. These parameters control the training process, model architecture, optimizer settings, and system configuration.

`train_config.yaml` defines the parameter defaults; these defaults are designed to train a GPT2 (124M) on OpenWebText.

## Usage Examples

### Override Config via Command Line

You can override any parameter from the command line:

```bash
# Change batch size and disable compilation
python train.py batch_size=128 compile=False

# Use a different learning rate and dataset
python train.py learning_rate=1e-3 dataset=openwebtext

# Train a smaller model
python train.py n_layer=6 n_head=6 n_embd=384

# Enable wandb logging with custom project name
python train.py wandb_log=True wandb_project=my-project wandb_run_name=experiment-1
```

### Multiple Overrides

```bash
python train.py batch_size=32 learning_rate=1e-3 max_iters=100000 compile=False
```

## Table of Contents

- [Output and Logging](#output-and-logging)
- [Weights and Biases (wandb) Logging](#weights-and-biases-wandb-logging)
- [Data Configuration](#data-configuration)
- [Model Architecture](#model-architecture)
- [AdamW Optimizer](#adamw-optimizer)
- [Learning Rate Decay Settings](#learning-rate-decay-settings)
- [Distributed Data Parallel (DDP) Settings](#distributed-data-parallel-ddp-settings)
- [System Settings](#system-settings)

## Output and Logging

### `out_dir`
- **Type**: `string`
- **Default**: `'out'`
- **Description**: Output directory where checkpoints and logs will be saved. The training script will create this directory if it doesn't exist.

### `log_interval`
- **Type**: `integer`
- **Default**: `1`
- **Description**: How often (in iterations) to log training loss and metrics to the console. A value of `1` means logging every iteration, while higher values reduce logging frequency.

### `eval_interval`
- **Type**: `integer`
- **Default**: `2000`
- **Description**: How often (in iterations) to evaluate the model on the validation set. Evaluation computes validation loss and can trigger checkpoint saving.
- **Example**: `eval_interval: 5000` (evaluate every 5000 iterations)

### `eval_iters`
- **Type**: `integer`
- **Default**: `200`
- **Description**: Number of batches to use when estimating validation loss. Higher values provide more accurate loss estimates but take longer to compute.

### `eval_only`
- **Type**: `boolean`
- **Default**: `False`
- **Description**: If `True`, the script will run a single evaluation and exit without training. Useful for evaluating a trained model or checking data loading.

### `always_save_checkpoint`
- **Type**: `boolean`
- **Default**: `True`
- **Description**: If `True`, a checkpoint will be saved after every evaluation, regardless of whether validation loss improved. If `False`, only save when validation loss decreases.

### `init_from`
- **Type**: `string`
- **Default**: `'scratch'`
- **Description**: Initialization strategy for the model. Options:
  - `'scratch'`: Initialize a new model from scratch with random weights
  - `'resume'`: Resume training from a checkpoint in `out_dir` (loads `ckpt.pt`)
  - `'gpt2'`, `'gpt2-medium'`, `'gpt2-large'`, `'gpt2-xl'`: Initialize from OpenAI's pretrained GPT-2 weights

## Weights and Biases (wandb) Logging

### `wandb_log`
- **Type**: `boolean`
- **Default**: `False`
- **Description**: Enable or disable Weights and Biases logging. When enabled, training metrics, losses, and hyperparameters will be logged to wandb.

### `wandb_project`
- **Type**: `string`
- **Default**: `'austen-gpt'`
- **Description**: Name of the wandb project where runs will be logged. All runs with the same project name will appear together in the wandb dashboard.

### `wandb_run_name`
- **Type**: `string`
- **Default**: `'gpt2-austen-12layer'`
- **Description**: Name for this specific training run in wandb. Useful for distinguishing between different experiments.

## Data Configuration

### `dataset`
- **Type**: `string`
- **Default**: `'austen'`
- **Description**: Name of the dataset to use. The script will look for preprocessed data files at `data/{dataset}/train.bin` and `data/{dataset}/val.bin`. These should be binary files containing tokenized data.
- **Example**: `dataset: 'openwebtext'` (for OpenWebText dataset)

### `batch_size`
- **Type**: `integer`
- **Default**: `12`
- **Description**: Minibatch size - the number of sequences processed in parallel during each forward pass. Larger batches require more GPU memory.

### `block_size`
- **Type**: `integer`
- **Default**: `1024`
- **Description**: Context window size - the maximum sequence length the model can process. This determines how many tokens the model can "see" at once. Larger values require more memory.

### `gradient_accumulation_steps`
- **Type**: `integer`
- **Default**: `40`
- **Description**: Number of gradient accumulation steps. This allows simulating larger batch sizes by accumulating gradients over multiple micro-batches before updating weights.
    - **Effective batch size = batch_size * gradient_accumulation_steps**
- **Why?** GPU memory can't fit the effective batch size, so we:
    1. forward/backward with minibatch (batch_size=12) 40 times
    2. accumulate gradients
    3. update weights once

## Model Architecture

### `n_layer`
- **Type**: `integer`
- **Default**: `12`
- **Description**: Number of transformer layers (blocks) in the model. More layers increase model capacity and training time. GPT-2 small uses 12 layers, GPT-2 medium uses 24, GPT-2 large uses 36.

### `n_head`
- **Type**: `integer`
- **Default**: `12`
- **Description**: Number of attention heads per transformer block. Must evenly divide `n_embd` (i.e., `n_embd % n_head == 0`). More heads allow the model to attend to different aspects of the input simultaneously.

### `n_embd`
- **Type**: `integer`
- **Default**: `768`
- **Description**: Embedding dimension - the size of the hidden state throughout the model. Larger values increase model capacity and memory usage. GPT-2 small uses 768, GPT-2 medium uses 1024, GPT-2 large uses 1280.

### `dropout`
- **Type**: `float`
- **Default**: `0.0`
- **Description**: Dropout rate applied throughout the model. Dropout helps prevent overfitting by randomly zeroing some activations during training. For pretraining, `0.0` is typically used. For finetuning, values like `0.1` or `0.2` can help.

### `bias`
- **Type**: `boolean`
- **Default**: `False`
- **Description**: Whether to use bias terms in Linear layers and LayerNorm layers. Setting to `False` (no bias) is slightly faster and often performs as well or better than using bias.

## AdamW Optimizer

### `learning_rate`
- **Type**: `float`
- **Default**: `6e-4` (0.0006)
- **Description**: Maximum learning rate during training. This is the peak learning rate reached after warmup. The actual learning rate may be lower due to warmup and decay schedules.

### `max_iters`
- **Type**: `integer`
- **Default**: `600000`
- **Description**: Total number of training iterations. Training will stop after this many iterations. One iteration processes one batch (or accumulated gradients).

### `weight_decay`
- **Type**: `float`
- **Default**: `1e-1` (0.1)
- **Description**: L2 regularization strength for weight decay in AdamW. Higher values penalize large weights more, helping prevent overfitting. This is a key hyperparameter for training stability.

### `beta1`
- **Type**: `float`
- **Default**: `0.9`
- **Description**: First moment decay rate for the AdamW optimizer. Controls the exponential decay rate for the first moment estimates (momentum-like term). 0.9 is the standard value that is rarely changed.

### `beta2`
- **Type**: `float`
- **Default**: `0.95`
- **Description**: Second moment decay rate for the AdamW optimizer. Controls the exponential decay rate for the second moment estimates (adaptive learning rate term). 0.95 is the standard value that is rarely changed.

### `grad_clip`
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Gradient clipping threshold. Gradients with norm exceeding this value will be clipped to prevent exploding gradients. Set to `0.0` to disable gradient clipping.

## Learning Rate Decay Settings

### `decay_lr`
- **Type**: `boolean`
- **Default**: `True`
- **Description**: Whether to use learning rate decay. If `True`, the learning rate will follow a cosine decay schedule with warmup. If `False`, the learning rate stays constant at `learning_rate`.

### `warmup_iters`
- **Type**: `integer`
- **Default**: `2000`
- **Description**: Number of iterations for the learning rate warmup phase. During warmup, the learning rate linearly increases from near zero to `learning_rate`. This helps stabilize early training.

### `lr_decay_iters`
- **Type**: `integer`
- **Default**: `600000`
- **Description**: Number of iterations over which the learning rate decays. Should typically equal `max_iters` per the Chinchilla scaling laws. The learning rate follows a cosine decay from `learning_rate` to `min_lr` over this period.

### `min_lr`
- **Type**: `float`
- **Default**: `6e-5` (0.00006)
- **Description**: Minimum learning rate reached at the end of training. According to Chinchilla scaling laws, this should be approximately `learning_rate / 10`.

## Distributed Data Parallel (DDP) Settings

### `backend`
- **Type**: `string`
- **Default**: `'nccl'`
- **Description**: Distributed training backend for multi-GPU training. Options:
  - `'nccl'`: NVIDIA Collective Communications Library (recommended for NVIDIA GPUs)
  - `'gloo'`: Alternative backend, works on CPU and some GPU setups

**Note**: DDP is automatically detected when using `torchrun`. The `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` environment variables control DDP behavior.

## System Settings

### `device`
- **Type**: `string`
- **Default**: `'cuda'`
- **Description**: Device to use for training. Options:
  - `'cpu'`: Use CPU (slow, for debugging)
  - `'cuda'`: Use default CUDA device
  - `'cuda:0'`, `'cuda:1'`, etc.: Use specific GPU
  - `'mps'`: Use Apple Silicon GPU (M1/M2 Macs)

### `dtype`
- **Type**: `string`
- **Default**: `'bfloat16'`
- **Description**: Data type for model computations. Options:
  - `'float32'`: Full precision (slower, more memory, most stable)
  - `'bfloat16'`: Brain floating point (faster, less memory, good for training)
  - `'float16'`: Half precision (fastest, least memory, may be less stable)
  
  **Note**: If `bfloat16` is specified but not supported by the hardware, the script automatically falls back to `float16`.

### `compile`
- **Type**: `boolean`
- **Default**: `True`
- **Description**: Whether to use PyTorch 2.0's `torch.compile()` to optimize the model. This can significantly speed up training (often 20-30% faster) but requires PyTorch 2.0+ and adds a compilation step at startup.

## Notes

- All paths are relative to the directory where `train.py` is executed
- When using DDP with `torchrun`, the `device` setting is automatically overridden to use the appropriate GPU for each process
- The effective batch size is `batch_size * gradient_accumulation_steps * num_gpus` (if using DDP)
- Model checkpoints include the full configuration, so you can resume training with different hyperparameters (though some model architecture parameters must match)

