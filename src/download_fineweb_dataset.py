from datasets import load_dataset
import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import tiktoken
import shutil

shard_size = int(1e8)
tokenizer = tiktoken.get_encoding('gpt2')
eot_token = tokenizer._special_tokens['<|endoftext|>']
data = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train')
dataset_folder = os.path.join(os.path.dirname(__file__), '../resources/edu_fineweb')
# Create dataset folder and subfolders for train and val splits
if os.path.exists(dataset_folder):
    shutil.rmtree(dataset_folder)
os.makedirs(dataset_folder)

train_folder = os.path.join(dataset_folder, 'train')
val_folder = os.path.join(dataset_folder, 'val')

os.makedirs(train_folder)
os.makedirs(val_folder)


def tokenize(doc):
    tokens = tokenizer.encode_ordinary(doc["text"])
    tokens.append(eot_token)
    return np.array(tokens).astype(np.uint16)


def write_datafile(split, dataset_folder, data, skip_first):
    shard_index, token_count = 0, 0
    current_shard = np.empty((shard_size,), dtype=np.uint16)
    progress_bar = None
    nprocs = max(1, os.cpu_count()-1)
    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenize, data, chunksize=16):
            current_tokens_len = len(tokens)
            current_tokens_len = len(tokens)
            if token_count + current_tokens_len < shard_size:
                current_shard[token_count:token_count + current_tokens_len] = tokens
                token_count += current_tokens_len
                if not progress_bar:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(current_tokens_len)
            else:
                filename = os.path.join(dataset_folder, f'edu_fw_{split}_{shard_index:06d}')
                remaining = shard_size - token_count
                current_shard[token_count:] = tokens[:remaining]
                np.save(filename, current_shard)
                shard_index += 1

                if skip_first:  # Skip first in test, dirty but works
                    skip_first = False
                    shard_index = 0

                token_count = current_tokens_len - remaining
                progress_bar = None
                current_shard = np.empty((shard_size,), dtype=np.uint16)
                current_shard[0:token_count] = tokens[remaining:]

                if split == 'val':
                    break

                if token_count != 0:
                    filename = os.path.join('{dataset_folder}/{split}', split, f'edu_fw_{split}_{shard_index:06d}')
                    np.save(filename, tokens[:token_count])


write_datafile('train', dataset_folder, data, True)
write_datafile('val', dataset_folder, data, False)
