import tiktoken
import torch


class ToyDataLoader:

    def __init__(self, data_path='./resources/input.txt', batch_size=4, max_seq_len=32):
        self.encodig = tiktoken.get_encoding('gpt2')
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def load_dataset(self, test_train_split):
        with open(self.data_path, 'r') as f:
            self.dataset = self.encodig.encode(f.read())
            self.dataset_len = len(self.dataset)
            self.start = 0
            self.start_valid = 0

        if test_train_split:
            test_size = int(self.dataset_len * test_train_split)
            indices = torch.randperm(self.dataset_len).tolist()

            self.valid_dataset = [self.dataset[i] for i in indices[:test_size] if i < self.dataset_len]
            self.train_data = [self.dataset[i] for i in indices[test_size:] if i < self.dataset_len]

    def next_valid_batch(self):

        chunk_size = min((self.batch_size * self.max_seq_len) + 1, len(self.valid_dataset)-1)
        if self.start_valid + chunk_size > self.dataset_len:
            self.start_valid = 0

        chunk = self.valid_dataset[self.start_valid:self.start_valid+chunk_size]
        self.start += chunk_size+1

        x = torch.tensor(chunk[:-1], dtype=torch.long)\
            .view(self.batch_size, self.max_seq_len)

        y = torch.tensor(chunk[1:], dtype=torch.long)\
            .view(self.batch_size, self.max_seq_len)

        return x, y
