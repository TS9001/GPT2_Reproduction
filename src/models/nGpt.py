import math
import torch
import torch.nn as nn
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss, LigerSwiGLUMLP
from liger_kernel.transformers.rope import LigerRopeFunction
from models.model_configuration import ModelConfiguration, TrainedNetwork
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from torch.nn import functional as F

class Attention(TrainedNetwork):
    def __init__(self, config: ModelConfiguration):
        super().__init__(config)
        self.c_attn = nn.Linear(config.d_model, config.d_model*3)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.num_heads = self.config.num_heads
        self.rope_dtype = config.rope_dtype
        cos, sin = self.precompute_freqs_cis(config.d_model // self.num_heads, config.block_size)
        self.register_buffer("cos_sp",  torch.cat([cos, cos], dim=-1).unsqueeze(0))  # [1, seq, dim]
        self.register_buffer("sin_sp",  torch.cat([sin, sin], dim=-1).unsqueeze(0))  # [1, seq, dim]
        self.attn_scale = math.sqrt(config.d_model / config.num_heads)
        self.sqk_init_value = 1.0
        self.sqk_init_scaling = config.base_scale
        self.sqk = torch.nn.Parameter(self.sqk_init_scaling*torch.ones(self.config.d_model // self.config.num_heads, dtype=torch.float32))


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
        sqk_scaled = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(1, 1, 1, self.config.d_model // self.config.num_heads )
        q_norm = sqk_scaled * _normalize_weights(q_bhsd)
        k_norm = sqk_scaled * _normalize_weights(k_bhsd)
        y_bhsd = torch.nn.functional.scaled_dot_product_attention(q_norm, k_norm, v_bhsd, is_causal=True, scale=self.attn_scale)
        y_bsd = self.c_proj(y_bhsd.transpose(1, 2).contiguous().view(B, S, D))
        return y_bsd

class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfiguration):
        super().__init__()
        self.config = config

        # Keep all your original layers
        self.w1 = nn.Linear(config.d_model, config.d_model * 4, bias=False)
        self.w2 = nn.Linear(config.d_model * 4, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_model * 4, bias=False)
        self.w2.NANOGPT_SCALE_INIT = 1

        # Scaling parameter
        self.suv_init_value = 1.0
        self.suv_init_scaling = config.base_scale
        self.suv = torch.nn.Parameter(self.suv_init_scaling*torch.ones(2 * 4 * config.d_model, dtype=torch.float32))

    def forward(self, x_bsd):
        # Calculate scaling factor
        suv = (self.suv * ((self.suv_init_value/self.suv_init_scaling) * (self.config.d_model ** 0.5)))
        su, sv = torch.chunk(suv, 2, dim=-1)
        w1_out = su * self.w1(x_bsd)
        w3_out = sv * self.w3(x_bsd)

        return self.w2(F.silu(w1_out) * w3_out)


class Normalized_LigerSwiGLUMLP(LigerSwiGLUMLP):
    def __init__(self, liger_config, config: ModelConfiguration):
        super().__init__(liger_config)
        self.config = liger_config
        self.suv_init_value = 1.0
        self.suv_init_scaling = config.base_scale
        self.suv = torch.nn.Parameter(self.suv_init_scaling * torch.ones(2 * 4 * config.d_model, dtype=torch.float32))

    def forward(self, x):
        # Simple scaling calculation
        scale = self.suv_init_value / self.suv_init_scaling
        gate_s, up_s = torch.chunk(self.suv * scale, 2, dim=-1)

        # Apply scaling to projections
        gate_out = gate_s * self.gate_proj(x)
        up_out = up_s * self.up_proj(x)

        # Apply SwiGLU activation and down projection
        hidden_states = LigerSiLUMulFunction.apply(gate_out, up_out)
        return self.down_proj(hidden_states)

class Block(nn.Module):
    def __init__(self, config: ModelConfiguration):
        super().__init__()
        self.config = config
        self.attn = Attention(config)
        if config.use_liger:
            liger_config = {
                'hidden_size': config.d_model,
                'intermediate_size': config.d_model * 4,
                'hidden_act': "silu"
            }
            self.mlp = Normalized_LigerSwiGLUMLP(type('Config', (), liger_config)(), config)
        else:
            self.mlp = SwiGLU(config)

        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = config.base_scale
        self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling*torch.ones(self.config.d_model, dtype=torch.float32))

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = config.base_scale
        self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling*torch.ones(self.config.d_model, dtype=torch.float32))

    def forward(self, h):
        attn_alpha = torch.abs(self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling))
        mlp_alpha = torch.abs(self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling))
        h_norm = _normalize_weights(h)
        h =  _normalize_weights(h_norm +  attn_alpha * (_normalize_weights(self.attn(h)) - h_norm))
        h = _normalize_weights(h + mlp_alpha * (_normalize_weights(self.mlp(h)) - h))
        return h


class ModelBasis(nn.Module):
    def __init__(self, config: ModelConfiguration):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.d_model),
            h=nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
        ))
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # non standard for t++ but saves some memory
        self.apply(self._init_weights)

        self.sz_init_value = 1.00
        self.sz_init_scaling = config.base_scale
        self.sz = torch.nn.Parameter(self.sz_init_scaling*torch.ones(config.vocab_size, dtype=torch.float32))


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
        sz = self.sz * (self.sz_init_value/self.sz_init_scaling)
        logits = sz * self.lm_head(x_bsd)
        loss = None

        if y is None:
            return logits, None

        loss = nn.functional.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            y.view(-1)
        )
        return logits if return_logits else None, loss

    def forward(self, x_bs, y=None, return_logits=True):  # [batch,seq]
        _, S = x_bs.size()
        seq_s = torch.arange(0, S, device=x_bs.device, dtype=torch.long)
        x_bsd = self.transformer.wte(x_bs)

        for block in self.transformer.h:
            x_bsd = block(x_bsd)

        return self.compute_standard_loss(x_bsd, y, return_logits)

    def post_training_step(self):
        self._post_step_normalize_weights()

    def _post_step_normalize_weights(self):
        self.lm_head.weight.data.copy_(_normalize_weights(self.lm_head.weight.data))

        for block in self.transformer.h:

            q, k, v = block.attn.c_attn.weight.data.split(self.config.d_model, dim=0)
            q = _normalize_weights(q)
            k = _normalize_weights(k)
            v = _normalize_weights(v)
            block.attn.c_attn.weight.data.copy_(torch.cat([q, k, v], dim=0))

            if self.config.use_liger:
                block.mlp.gate_proj.weight.data.copy_(_normalize_weights(block.mlp.gate_proj.weight.data))
                block.mlp.up_proj.weight.data.copy_(_normalize_weights(block.mlp.up_proj.weight.data))
                block.mlp.down_proj.weight.data.copy_(_normalize_weights(block.mlp.down_proj.weight.data))
            else:
                block.mlp.w1.weight.data.copy_(_normalize_weights(block.mlp.w1.weight.data))
                block.mlp.w2.weight.data.copy_(_normalize_weights(block.mlp.w2.weight.data))
                block.mlp.w3.weight.data.copy_(_normalize_weights(block.mlp.w3.weight.data))

def _normalize_weights(x, norm_dim = -1, eps=1e-8):
    orig_type = x.dtype
    x = x.to(torch.float32)
    return (x / (x.norm(p=2, dim=norm_dim, keepdim=True) + eps)).to(orig_type)