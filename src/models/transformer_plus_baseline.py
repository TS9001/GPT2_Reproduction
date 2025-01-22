import torch
import torch.nn as nn
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss, LigerRMSNorm, LigerSwiGLUMLP
from liger_kernel.transformers.rope import LigerRopeFunction
from models.model_configuration import ModelConfiguration
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, config: ModelConfiguration):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.d_model, config.d_model*3)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.num_heads = self.config.num_heads
        self.rope_dtype = config.rope_dtype
        cos, sin = self.precompute_freqs_cis(config.d_model // self.num_heads, config.block_size)
        self.register_buffer("cos_sp",  torch.cat([cos, cos], dim=-1).unsqueeze(0))  # [1, seq, dim]
        self.register_buffer("sin_sp",  torch.cat([sin, sin], dim=-1).unsqueeze(0))  # [1, seq, dim]

    def precompute_freqs_cis(self, dim: int, end: int,  theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim , 2, dtype=self.rope_dtype) / (dim)))
        freqs = torch.outer(torch.arange(end,  dtype=self.rope_dtype), freqs)  # [end, dim/2]
        return torch.cos(freqs), torch.sin(freqs)

    def apply_rotary_position_embedding(self, q, k,  cos, sin):
        q_ri =  torch.cat((-q[..., q.shape[-1] // 2:], q[..., :q.shape[-1] // 2]), dim=-1)
        k_ri =  torch.cat((-k[..., k.shape[-1] // 2:], k[..., :k.shape[-1] // 2]), dim=-1)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_rotated = (q * cos) + (q_ri * sin)
        k_rotated = (k * cos) + (k_ri * sin)
        return q_rotated.to(q.dtype), k_rotated.to(k.dtype)

    def forward(self, x_bsd):  # [batch,seq,dim]
        B, S, D = x_bsd.size()
        head_dim = D // self.num_heads

        qkv_bhsd = map(lambda t: t.view(B, S, self.num_heads, head_dim).transpose(1, 2),
                       self.c_attn(x_bsd).chunk(3, dim=-1))
        q_bhsd, k_bhsd, v_bhsd = qkv_bhsd

        if not self.config.use_liger:
            q_bhsd, k_bhsd = self.apply_rotary_position_embedding(q_bhsd, k_bhsd, self.cos_sp[:, :S], self.sin_sp[:, :S])
        else:
                q_bhsd, k_bhsd = LigerRopeFunction.apply(
                    q_bhsd, k_bhsd,
                    self.cos_sp[:, :S], self.sin_sp[:, :S],
                    None,
                    1     # unsqueeze_dim
                )

        y_bhsd = torch.nn.functional.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, is_causal=True)
        y_bsd = self.c_proj(y_bhsd.transpose(1, 2).contiguous().view(B, S, D))
        return y_bsd


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight_d = nn.Parameter(torch.ones(dim))

    def forward(self, x_bsd):
        return x_bsd * torch.rsqrt((x_bsd * x_bsd).mean(-1, keepdim=True) + self.eps) * self.weight_d


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfiguration):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.d_model, config.d_model * 4, bias=False)
        self.w2 = nn.Linear(config.d_model * 4, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_model * 4, bias=False)
        self.w2.NANOGPT_SCALE_INIT = 1

    def forward(self, x_bsd):
            return self.w2(F.silu(self.w1(x_bsd)) * self.w3(x_bsd))


class Block(nn.Module):
    def __init__(self, config: ModelConfiguration):
        super().__init__()
        self.config = config
        self.ln_1 = LigerRMSNorm(config.d_model) if config.use_liger else RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.ln_2 = LigerRMSNorm(config.d_model) if config.use_liger else RMSNorm(config.d_model)
        if config.use_liger:
            liger_config = ModelConfiguration(
                hidden_size=config.d_model,
                intermediate_size=config.d_model * 4,
                hidden_act="silu"
            )
            self.mlp = LigerSwiGLUMLP(liger_config)
        else:
            self.mlp = SwiGLU(config)

    def forward(self, x_bsd):
        x_bsd = x_bsd + self.attn(self.ln_1(x_bsd))
        x_bsd = x_bsd + self.mlp(self.ln_2(x_bsd))
        return x_bsd


class GPT2Basic(nn.Module):
    def __init__(self, config: ModelConfiguration):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.d_model),
            wpe=nn.Embedding(config.block_size, config.d_model),
            h=nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f=LigerRMSNorm(config.d_model) if config.use_liger else RMSNorm(config.d_model)
        ))
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
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

    def forward(self, x_bs, y=None, return_logits=True):  # [batch,seq]
        _, S = x_bs.size()
        seq_s = torch.arange(0, S, device=x_bs.device, dtype=torch.long)
        x_bsd = self.transformer.wte(x_bs) + self.transformer.wpe(seq_s)

        for block in self.transformer.h:
            x_bsd = block(x_bsd)

        x_bsd = self.transformer.ln_f(x_bsd)

        return self.compute_standard_loss(x_bsd, y, return_logits) if not self.config.use_liger else self.compute_liger_loss(x_bsd, y, return_logits)
