import torch
import torch.nn as nn

class GPT2Configuration:
    def __init__(
        self,
        block_size: int = 1024,
        num_layers: int = 12,
        num_heads: int = 12,
        d_model: int = 768,
        vocab_size: int = 50304,
        use_liger: bool = False
    ):
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_heads = num_heads
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_liger = use_liger


class Attention(nn.Module):

    def __init__(self, config: GPT2Configuration):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.d_model, config.d_model*3)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.register_buffer(
            'bias',
            torch.tril(
                torch.ones(config.block_size, config.block_size).view(
                    1, 1, config.block_size, config.block_size)
            )
        )

        self.num_heads = self.config.num_heads

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.config.num_heads, d_model //
                      self.config.num_heads).transpose(1, 2), self.c_attn(x).chunk(3, dim=-1))

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.c_proj(y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model))


class MLP(nn.Module):
    def __init__(self, config: GPT2Configuration):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_model*4)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(config.d_model*4, config.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPT2Configuration):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Basic(nn.Module):
    def __init__(self, config: GPT2Configuration):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.d_model),
            wpe=nn.Embedding(config.block_size, config.d_model),
            h=nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f=nn.LayerNorm(config.d_model)
        ))
        self.lm_head = torch.nn.Linear(
            config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None, return_logits=True):
        _, seq_len = x.size()
        loss = None
        sequence = torch.arange(0, seq_len, device=x.device, dtype=torch.long)
        x = self.transformer.wte(x) + self.transformer.wpe(sequence)

        for layer in self.transformer.h:
            x = layer(x)

        x = self.transformer.ln_f(x)
        logits = None
        if return_logits:
            logits = self.lm_head(x)

        if y is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1))

        return logits, loss
