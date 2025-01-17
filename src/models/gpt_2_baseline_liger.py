import torch
import torch.nn as nn
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss, LigerRMSNorm
from liger_kernel.transformers.rope import LigerRopeFunction

class GPT2Configuration:
    # dimension suffixes: b = batch, h = heads, s = sequence, d = dimension, p = pairs, v = vocab
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
        self.num_heads = self.config.num_heads
        freqs_cis_sp = self.precompute_freqs_cis(config.d_model // self.num_heads, config.block_size)

        if not config.use_liger:
            self.register_buffer( 'freqs_cis_sp', freqs_cis_sp) # [seq, pairs]
        else:
            self.register_buffer("cos_sp",  torch.cat([freqs_cis_sp.real,  freqs_cis_sp.real], dim=-1).unsqueeze(0))  # [1, seq, dim]
            self.register_buffer("sin_sp",  torch.cat([freqs_cis_sp.imag , freqs_cis_sp.imag], dim=-1).unsqueeze(0))  # [1, seq, dim]

    def precompute_freqs_cis(self, dim: int, end: int,  theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim , 2, dtype=torch.float32) / (dim)))
        t = torch.arange(end,  dtype=torch.float32)
        freqs = torch.outer(t, freqs)  # [end, dim/2]
        freqs_cis = torch.complex(torch.cos(freqs), torch.sin(freqs))

        return freqs_cis

    def apply_rotary_position_embedding(self, q, k, freqs_cis_sp):  # [batch,heads,seq,dim]
        # Reorder q, k so that their last dim is [half, 2]

        q_ri =  torch.stack((q[..., :q.shape[-1] // 2], q[..., q.shape[-1] // 2:]), dim=-1)
        k_ri =  torch.stack((k[..., :k.shape[-1] // 2], k[..., k.shape[-1] // 2:]), dim=-1)
        q_complex = torch.view_as_complex(q_ri.float())  # shape [batch, heads, seq, half]
        k_complex = torch.view_as_complex(k_ri.float())
        # Multiply => apply rotation
        q_rotated = q_complex * freqs_cis_sp   # shape [batch, heads, seq, half]
        k_rotated = k_complex * freqs_cis_sp

        # Convert back to real
        q_ri_out = torch.view_as_real(q_rotated)  # shape [..., half, 2]
        k_ri_out = torch.view_as_real(k_rotated)

        # "Unstack" them so that the last dimension is 2*half = dim
        # which is [real, imag] => cat => [..., dim]
        q_out = torch.cat([q_ri_out[..., 0], q_ri_out[..., 1]], dim=-1)
        k_out = torch.cat([k_ri_out[..., 0], k_ri_out[..., 1]], dim=-1)

        return q_out, k_out

    def forward(self, x_bsd):  # [batch,seq,dim]
        B, S, D = x_bsd.size()
        head_dim = D // self.num_heads

        qkv_bhsd = map(lambda t: t.view(B, S, self.num_heads, head_dim).transpose(1, 2),
                       self.c_attn(x_bsd).chunk(3, dim=-1))
        q_bhsd, k_bhsd, v_bhsd = qkv_bhsd

        if not self.config.use_liger:
            q_bhsd, k_bhsd = self.apply_rotary_position_embedding(q_bhsd, k_bhsd, self.freqs_cis_sp[:S])
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

    def reinit_rope_buffers(self):
        """Reinitialize ROPE buffers based on current config"""
        freqs_cis_sp = self.precompute_freqs_cis(self.config.d_model // self.num_heads, self.config.block_size)

        # Remove existing buffers
        self._buffers.pop('freqs_cis_sp', None)
        self._buffers.pop('cos_sp', None)
        self._buffers.pop('sin_sp', None)

        # Register appropriate buffers based on config
        if not self.config.use_liger:
            self.register_buffer('freqs_cis_sp', freqs_cis_sp)
        else:
            self.register_buffer("cos_sp", torch.cat([freqs_cis_sp.real, freqs_cis_sp.real], dim=-1).unsqueeze(0))
            self.register_buffer("sin_sp", torch.cat([freqs_cis_sp.imag, freqs_cis_sp.imag], dim=-1).unsqueeze(0))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight_d = nn.Parameter(torch.ones(dim))

    def forward(self, x_bsd):
        return x_bsd * torch.rsqrt((x_bsd * x_bsd).mean(-1, keepdim=True) + self.eps) * self.weight_d


class MLP(nn.Module):
    def __init__(self, config: GPT2Configuration):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_model*4)
        self.act = nn.SiLU()
        self.c_proj = nn.Linear(config.d_model*4, config.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x_bsd):
        return self.c_proj(self.act(self.c_fc(x_bsd)))


class Block(nn.Module):
    def __init__(self, config: GPT2Configuration):
        super().__init__()
        self.config = config
        self.ln_1 = LigerRMSNorm(config.d_model) if config.use_liger else RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.ln_2 = LigerRMSNorm(config.d_model) if config.use_liger else RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x_bsd):
        x_bsd = x_bsd + self.attn(self.ln_1(x_bsd))
        x_bsd = x_bsd + self.mlp(self.ln_2(x_bsd))
        return x_bsd


class GPT2Basic(nn.Module):
    def __init__(self, config: GPT2Configuration):
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
