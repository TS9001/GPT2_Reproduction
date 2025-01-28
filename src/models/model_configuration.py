import torch
from torch import dtype
from torch import nn
class ModelConfiguration:
    def __init__(
        self,
        block_size: int = 1024,
        num_layers: int = 12,
        num_heads: int = 12,
        d_model: int = 768,
        vocab_size: int = 50304,
        use_liger: bool = False,
        rope_dtype: dtype = torch.float32
    ):
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_heads = num_heads
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_liger = use_liger
        self.rope_dtype = rope_dtype

class TrainedNetwork(nn.Module):
    def __init__(self, config: ModelConfiguration):
        super().__init__()
        self.config = config

    def post_training_step(self):
        pass

    def pre_training_step(self):
        pass