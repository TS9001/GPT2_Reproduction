import torch
import torch.nn as nn
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss, LigerLayerNorm
from models.model_configuration import ModelConfiguration
class Attention(nn.Module):

    def __init__(self, config: ModelConfiguration):
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
    def __init__(self, config: ModelConfiguration):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_model*4)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(config.d_model*4, config.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfiguration):
        super().__init__()
        self.config=config

        self.ln_1=LigerLayerNorm(config.d_model) if self.config.use_liger else nn.LayerNorm(config.d_model)
        self.attn=Attention(config)
        self.ln_2=LigerLayerNorm(config.d_model) if self.config.use_liger else nn.LayerNorm(config.d_model)
        self.mlp=MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Basic(nn.Module):
    def __init__(self, config: ModelConfiguration):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.d_model),
            wpe=nn.Embedding(config.block_size, config.d_model),
            h=nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f=LigerLayerNorm(config.d_model) if self.config.use_liger else nn.LayerNorm(config.d_model)
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

        return self.compute_standard_loss(x, y, return_logits) if not self.config.use_liger else self.compute_liger_loss(x, y, return_logits)

    def compute_standard_loss(self, x_bsd, y, return_logits=True):
        logits = self.lm_head(x_bsd)
        loss = None

        if y is None:
            return logits, None

        loss = nn.functional.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            y.view(-1)
        )
        return logits if return_logits else None, loss

    def compute_liger_loss(self, x_bsd, y, return_logits=True):
        if y is None:
            return self.lm_head(x_bsd), None

        loss = LigerFusedLinearCrossEntropyLoss()(
            self.lm_head.weight.to(torch.bfloat16),
            x_bsd.view(-1, x_bsd.size(-1)).to(torch.bfloat16),
            y.view(-1)
        )

        logits = self.lm_head(x_bsd) if return_logits else None
        return logits, loss