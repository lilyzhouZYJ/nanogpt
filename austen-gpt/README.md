# AustenGPT

This directory implements a simple GPT model, based on Andrej Karpathy's [NanoGPT](https://github.com/karpathy/nanoGPT/tree/master). We train our model on Jane Austen's texts, but it also supports training on OpenWebText.

## Table of Contents

- [Model Architecture](#model-architecture)
- [Training](#training)
  - [(1) Prepare data](#1-prepare-data)
  - [(2) Train](#2-train)
- [Text Generation](#text-generation)

## Model Architecture

Our model uses the Transformer architecture, shown in the below diagram. On the left is the architecture described in the "Attention is all you need" paper. We will omit the cross-attention block, so our actual architecture is shown by the **diagram on the right**, with one caveat: the convention now is to apply LayerNorm *before* the attention layer and the FFN layer, instead of after.

![architecture.pnb](/asset/architecture.png)

*Diagram source: https://www.ericjwang.com/assets/images/gpt_arch.png*

### Implementation

- **`block.py`**: implements a Transformer block
- **`gpt.py`**: implements the GPT model
- `config.py`: defines model configuration

### Documentation

For more details on the Transformer Blocks and the GPT model, see **[`docs/gpt.md`](/austen-gpt/docs/gpt.md)**.

## Training

**Documentation:**
- **[`docs/prepare_data.md`](/austen-gpt/docs/prepare_data.md)**
- **[`docs/training_config.md`](/austen-gpt/docs/training_config.md)**
- **[`docs/train.md`](/austen-gpt/docs/train.md)**

### (1) Prepare data

Before training the model, we first prepare the dataset; this includes downloading, processing, and tokenizing the text data. The preparation is done by the **`prepare_data.py`** scripts in the `data/{dataset}` directories. For example, [`data/austen/prepare_data.py`](/austen-gpt/data/austen/prepare_data.py).

To prepare the data, run:

```python
python data/austen/prepare_data.py
```

This will generate two binary files in the directory `austen-gpt/data/austen`:

- **`train.bin`**: contains tokenized training data (uint16 array)
- **`val.bin`**: contains containing tokenized validation data (uint16 array)

These files are used by `train.py` during training. The training script expects these files at:
- `data/{dataset}/train.bin`
- `data/{dataset}/val.bin`

### (2) Train

#### Train with default parameters:

```bash
python train.py
```

Default parameters are defined in **`train_config.yaml`**. These values are designed to train a GPT2 (124M) model on OpenWebText.

#### Train with custom parameters:

You may override default parameters from command line:

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

For example, to train on Jane Austen's Emma (230,203 tokens) on a single GPU, I ran:

```bash
python train.py \
    dtype=float16 \
    n_layer=2 \
    n_head=2 \
    n_embd=64 \
    batch_size=8 \
    block_size=64 \
    max_iters=1000 \
    learning_rate=3e-3 \
    eval_interval=20 \
    eval_iters=10 \
    dropout=0.1
```

Because the dataset is small, the model was also kept small to prevent overfitting.

## Text Generation