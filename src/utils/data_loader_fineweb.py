import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import torch

shard_size = int(1e8)


class FinewebEduDataset:

    def __init__(self, data, tokenizer, dataset_folder, batch_size=4, max_seq_len=32, split='train', process_rank=0, ):

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.process_rank = process_rank
        self.split = split
        self.data = data
        self.tokenizer = tokenizer
        self.eot_token = self.tokenizer._special_tokens['<|endoftext|>']
        self.dataset_folder = f'{dataset_folder}/{split}'

        if not os.path.exists(self.dataset_folder):
            os.makedirs(self.dataset_folder, exist_ok=True)
            self.write_datafile()

        self.shards = [shard for shard in os.listdir(self.dataset_folder)]
        self.reset()
        self.steps_per_shard = len(self.tokens) // (batch_size * max_seq_len)

    def reset(self):
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.batch_size * self.max_seq_len * self.process_rank

    def tokenize(self, doc):
        tokens = self.tokenizer.encode_ordinary(doc["text"])
        tokens.append(self.eot_token)
        return np.array(tokens).astype(np.uint16)

    def load_tokens(self, filename):
        npt = np.load(f'{self.dataset_folder}\\{filename}').astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)

        return ptt

    def write_datafile(self):
        shard_index, token_count = 0, 0
        skip_first = (self.split == 'train')
        current_shard = np.empty((shard_size,), dtype=np.uint16)
        progress_bar = None
        nprocs = max(1, os.cpu_count()//2)
        with mp.Pool(nprocs) as pool:
            for tokens in pool.imap(self.tokenize, self.data, chunksize=16):
                current_tokens_len = len(tokens)
                current_tokens_len = len(tokens)
                if token_count + current_tokens_len < shard_size:
                    current_shard[token_count:token_count + current_tokens_len] = tokens
                    token_count += current_tokens_len
                    if not progress_bar:
                        progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(current_tokens_len)
                else:
                    filename = os.path.join(self.dataset_folder, f'edu_fw_{self.split}_{shard_index:06d}')
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

                    if self.split == 'val':
                        break

                    if token_count != 0:
                        filename = os.path.join(self.dataset_folder, f'edu_fw_{self.split}_{shard_index:06d}')
                        np.save(filename, tokens[:token_count])

    def next_batch(self):
        batch_size, max_seq_len, process_rank = self.batch_size, self.max_seq_len, self.process_rank
        data = self.tokens[self.current_position:self.current_position + (max_seq_len*batch_size+1)]
        x, y = (data[:-1]).view(batch_size, max_seq_len), (data[1:]).view(batch_size, max_seq_len)
        self.current_position += batch_size * max_seq_len * process_rank
        if self.current_position + (batch_size * max_seq_len) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = batch_size * max_seq_len * process_rank

        x = x.pin_memory()
        y = y.pin_memory()
        return x, y

 # def write_datafile(self):
    #     shard_index, token_count = 0, 0
    #     current_shard = np.empty((shard_size,), dtype=np.uint16)
    #     progress_bar = None

    #     skip_first = (self.split == 'train')

    #     for doc in self.data:
    #         tokens = self.tokenize(doc)
    #         current_tokens_len = len(tokens)

    #         if token_count + current_tokens_len < shard_size:
    #             current_shard[token_count:token_count + current_tokens_len] = tokens
    #             token_count += current_tokens_len
    #             if not progress_bar:
    #                 progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
    #             progress_bar.update(current_tokens_len)
    #         else:
    #             filename = os.path.join(self.dataset_folder, f'edu_fw_{self.split}_{shard_index:06d}')
    #             remaining = shard_size - token_count
    #             current_shard[token_count:] = tokens[:remaining]
    #             np.save(filename, current_shard)
    #             shard_index += 1

    #             if skip_first:  # Skip first in test, dirty but works
    #                 skip_first = False
    #                 shard_index = 0

    #             token_count = current_tokens_len - remaining
    #             progress_bar = None
    #             current_shard = np.empty((shard_size,), dtype=np.uint16)
    #             current_shard[0:token_count] = tokens[remaining:]

    #             if self.split == 'val':  # Store only first in val
    #                 break

    #     if token_count != 0:
    #         filename = os.path.join(self.dataset_folder, f'edu_fw_{self.split}_{shard_index:06d}')
    #         np.save(filename, current_shard[:token_count])

    # def write_datafile(self):
    #     nprocs = max(1, os.cpu_count()//2)
    #     with mp.Pool(nprocs) as pool:
    #         shard_index, token_count = 0, 0
    #         current_shard = np.empty((shard_size,), dtype=np.uint16)
    #         progress_bar = None
    #         for tokens in pool.imap(self.tokenize, self.data, chunksize=16):
    #             current_tokens_len = len(tokens)
    #             if token_count + current_tokens_len < shard_size:
    #                 current_shard[token_count:token_count + len(tokens)] = tokens
    #                 token_count += current_tokens_len
    #                 if not progress_bar:
    #                     progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
    #                 progress_bar.update(len(tokens))
    #             else:
    #                 filename = os.path.join(self.dataset_folder, f'edu_fw_{self.split}_{shard_index:06d}')
    #                 remaining = shard_size - token_count
    #                 current_shard[token_count:] = tokens[:remaining]

    #                 if self.split != 'train' or shard_index != 0:
    #                     np.save(filename, current_shard)
    #                     shard_index += 1

    #                     token_count = current_tokens_len - remaining
    #                     current_shard = np.empty((shard_size,), dtype=np.uint16)
    #                     progress_bar = None
    #                     current_shard[0:current_tokens_len - remaining] = tokens[remaining:]

    #                 if self.split == 'val':
    #                     break

    #         if token_count != 0:
    #             filename = os.path.join(self.dataset_folder, f'edu_fw_{self.split}_{shard_index:06d}')
    #             np.save(filename, tokens[:token_count])
