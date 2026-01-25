# Data Preparation

Before training the model, we must first prepare the dataset; this includes downloading, processing, and tokenizing the text data.

The preparation is done by the `prepare_data.py` scripts in the `data/{dataset}` directories. For example, [data/austen/prepare_data.py](/austen-gpt/data/austen/prepare_data.py).

## Preparing the `austen` dataset

> Implementation: [data/austen/prepare_data.py](/austen-gpt/data/austen/prepare_data.py)

This script does the following:

1. Download the Austen dataset (if not already present)
2. Split the data into **training set** (first 90%) and **validation set** (last 10%)
3. Tokenize the text using **GPT-2's BPE (Byte Pair Encoding) tokenizer**
4. Export tokenized data to binary files (**`train.bin`** and **`val.bin`**); the training process will look for these files

### Tokenization: BPE (Byte Pair Encoding)

BPE is a subword tokenization scheme that:

- Breaks text into subword units (words, word parts, or characters for rare cases)
- Uses a vocabulary of ~50,000 tokens (GPT-2's vocabulary size)
- Is the same tokenization used by GPT-2, GPT-3, and other OpenAI models

### Output files

The script generates two binary files in the same directory as the script:

- **`train.bin`**: contains tokenized training data (uint16 array)
- **`val.bin`**: contains containing tokenized validation data (uint16 array)

These files are used by `train.py` during training. The training script expects these files at:
- `data/{dataset}/train.bin`
- `data/{dataset}/val.bin`

Where `{dataset}` is the value of the `dataset` config parameter (default: `'austen'`).

### Why export in binary format?

- Much faster to load than text files
- More memory efficient
- Can be memory-mapped for efficient random access during training
