import os
import requests
import tiktoken
import numpy as np

# Dataset
data_urls = [
    'https://raw.githubusercontent.com/lilyzhouZYJ/austen-gpt/main/dataset/emma.txt',
    'https://raw.githubusercontent.com/lilyzhouZYJ/austen-gpt/main/dataset/pride_and_prejudice.txt',
    'https://raw.githubusercontent.com/lilyzhouZYJ/austen-gpt/main/dataset/sense_and_sensibility.txt',
    'https://raw.githubusercontent.com/lilyzhouZYJ/austen-gpt/main/dataset/mansfield_park.txt',
    'https://raw.githubusercontent.com/lilyzhouZYJ/austen-gpt/main/dataset/persuasion.txt',
]

# Download and combine all files from data_urls into a single input.txt
script_dir = os.path.dirname(__file__)
input_file_path = os.path.join(script_dir, 'input.txt')
if not os.path.exists(input_file_path):
    # Ensure directory exists
    os.makedirs(script_dir, exist_ok=True)
    # Download and combine all files from URLs
    with open(input_file_path, 'w', encoding='utf-8') as outfile:
        for url in data_urls:
            print(f"Downloading {url}...")
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            outfile.write(response.text)
            outfile.write('\n\n')  # Add separation between files
    print(f"Combined dataset saved to {input_file_path}")

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Encode with tiktoken GPT-2 BPE
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(script_dir, 'train.bin'))
val_ids.tofile(os.path.join(script_dir, 'val.bin'))

# For Austen dataset:
# Train has 768,888 tokens
# Val has 83,742 tokens