import os
import numpy as np
import torch

shard_size = int(1e8)


class FinewebEduDataset:

    def __init__(self, dataset_folder, batch_size=4, max_seq_len=32, split='train', process_rank=0, ):

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.process_rank = process_rank
        self.split = split
        self.dataset_folder = f'{dataset_folder}/{split}'

        self.shards = [shard for shard in os.listdir(self.dataset_folder)]
        self.reset()
        self.steps_per_shard = len(self.tokens) // (batch_size * max_seq_len)

    def reset(self):
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.batch_size * self.max_seq_len * self.process_rank

    def load_tokens(self, filename):
        npt = np.load(f'{self.dataset_folder}/{filename}').astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)

        return ptt

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
