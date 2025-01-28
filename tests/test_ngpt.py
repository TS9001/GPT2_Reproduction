import torch
import pytest
from models.nGpt import (_normalize_weights, ModelBasis, Attention,
                        SwiGLU, Normalized_LigerSwiGLUMLP, Block)
from models.model_configuration import ModelConfiguration

def test_normalize_weights():
    # Test case 1: Simple 2D tensor
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    normalized = _normalize_weights(x)

    # Check shape remains the same
    assert normalized.shape == x.shape

    # Check each row is normalized (L2 norm should be approximately 1)
    row_norms = torch.norm(normalized, p=2, dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-6)

    # Test case 2: 1D tensor
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    normalized = _normalize_weights(x)
    norm = torch.norm(normalized, p=2)
    assert torch.isclose(norm, torch.tensor(1.0), atol=1e-6)

    # Test case 3: Test with different norm_dim
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    normalized = _normalize_weights(x, norm_dim=0)
    col_norms = torch.norm(normalized, p=2, dim=0)
    assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-6)

def test_model_weight_normalization():
    # Create a small model configuration for testing
    config = ModelConfiguration(
        num_layers=2,
        num_heads=2,
        d_model=32,
        vocab_size=100,
        block_size=32,
        use_liger=False,
        rope_dtype=torch.float32,
    )

    model = ModelBasis(config)

    # Store initial weights
    initial_wte = model.transformer.wte.weight.data.clone()
    initial_lm_head = model.lm_head.weight.data.clone()

    # Call post training step
    model.post_training_step()

    # Check that weights are normalized
    def check_weight_normalization(weight, dim=-1):
        norms = torch.norm(weight, p=2, dim=dim)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    # Check embedding and lm_head weights
    check_weight_normalization(model.transformer.wte.weight.data)
    check_weight_normalization(model.lm_head.weight.data)

    # Check attention weights in each block
    for block in model.transformer.h:
        # Check attention weights
        q, k, v = block.attn.c_attn.weight.data.split(config.d_model, dim=0)
        check_weight_normalization(q)
        check_weight_normalization(k)
        check_weight_normalization(v)

        # Check MLP weights
        check_weight_normalization(block.mlp.w1.weight.data)
        check_weight_normalization(block.mlp.w2.weight.data)
        check_weight_normalization(block.mlp.w3.weight.data)

    # Verify weights have actually changed from initial values
    assert not torch.allclose(initial_wte, model.transformer.wte.weight.data)
    assert not torch.allclose(initial_lm_head, model.lm_head.weight.data)

def test_normalize_weights_dtype_preservation():
    # Test that the function preserves input dtype
    dtypes = [torch.float16, torch.float32, torch.float64]

    for dtype in dtypes:
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        normalized = _normalize_weights(x)
        assert normalized.dtype == dtype

def test_attention_initialization():
    """Test that attention module is initialized correctly with paper's base_scale"""
    config = ModelConfiguration(
        num_layers=1,
        num_heads=2,
        d_model=32,
        vocab_size=50304,  # Paper's vocab size
        block_size=32,
        use_liger=False,
        rope_dtype=torch.float32
    )

    attn = Attention(config)

    # Test initial scaling parameters
    expected_base_scale = 1 / (50304 ** 0.5)
    assert torch.allclose(
        attn.sqk.data,
        torch.ones_like(attn.sqk.data) * expected_base_scale
    )
    assert attn.sqk_init_value == 1.0
    assert attn.sqk_init_scaling == expected_base_scale

def test_attention_scaling_computation():
    """Test that attention scaling is computed correctly"""
    config = ModelConfiguration(
        num_layers=1,
        num_heads=2,
        d_model=32,
        vocab_size=50304,  # Paper's vocab size
        block_size=32,
        use_liger=False,
        rope_dtype=torch.float32
    )

    attn = Attention(config)

    # Create sample input
    B, S, D = 2, 4, config.d_model  # batch, sequence, dimension
    x = torch.randn(B, S, D)

    # Get QKV projections
    qkv = attn.c_attn(x)
    q, k, v = qkv.chunk(3, dim=-1)

    # Reshape to heads
    head_dim = D // attn.num_heads
    q = q.view(B, S, attn.num_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, attn.num_heads, head_dim).transpose(1, 2)

    # Check that scaling is applied correctly
    sqk_scaled = attn.sqk * (attn.sqk_init_value / attn.sqk_init_scaling)
    sqk_scaled = sqk_scaled.view(1, 1, 1, -1)[:, :, :, :head_dim]  # Reshape to match head dimension
    q_norm = sqk_scaled * _normalize_weights(q)
    k_norm = sqk_scaled * _normalize_weights(k)

    # Verify shapes
    assert q_norm.shape == q.shape
    assert k_norm.shape == k.shape

    # Verify normalization
    assert torch.allclose(
        torch.norm(_normalize_weights(q), p=2, dim=-1),
        torch.ones(B, attn.num_heads, S),
        atol=1e-6
    )

def test_attention_forward():
    """Test the complete attention forward pass"""
    config = ModelConfiguration(
        num_layers=1,
        num_heads=2,
        d_model=32,
        vocab_size=50304,  # Paper's vocab size
        block_size=32,
        use_liger=False,
        rope_dtype=torch.float32
    )

    attn = Attention(config)

    # Test with different batch sizes and sequence lengths
    test_cases = [
        (1, 4),   # minimal case
        (2, 16),  # typical case
        (3, 32),  # full block size
    ]

    for batch_size, seq_len in test_cases:
        x = torch.randn(batch_size, seq_len, config.d_model)
        output = attn(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, config.d_model)

        # Check output is not None or NaN
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

def test_attention_scaling_gradients():
    """Test that gradients flow correctly through attention scaling"""
    config = ModelConfiguration(
        num_layers=1,
        num_heads=2,
        d_model=32,
        vocab_size=50304,  # Paper's vocab size
        block_size=32,
        use_liger=False,
        rope_dtype=torch.float32
    )

    attn = Attention(config)

    # Create input that requires grad
    x = torch.randn(2, 4, config.d_model, requires_grad=True)

    # Forward pass
    output = attn(x)

    # Create dummy loss and backward
    loss = output.sum()
    loss.backward()

    # Check that gradients exist and are not None
    assert x.grad is not None
    assert attn.sqk.grad is not None
    assert not torch.isnan(attn.sqk.grad).any()

def test_attention_rope():
    """Test rotary position embeddings in attention"""
    config = ModelConfiguration(
        num_layers=1,
        num_heads=2,
        d_model=32,
        vocab_size=50304,  # Paper's vocab size
        block_size=32,
        use_liger=False,
        rope_dtype=torch.float32
    )

    attn = Attention(config)

    # Check that RoPE buffers are properly initialized
    assert hasattr(attn, 'cos_sp')
    assert hasattr(attn, 'sin_sp')

    # Verify RoPE buffer shapes
    assert attn.cos_sp.shape[0] == 1  # [1, seq, dim]
    assert attn.cos_sp.shape[1] == config.block_size
    assert attn.sin_sp.shape == attn.cos_sp.shape

    # Test RoPE application
    q = torch.randn(2, attn.num_heads, 4, config.d_model // attn.num_heads)
    k = torch.randn(2, attn.num_heads, 4, config.d_model // attn.num_heads)

    q_rot, k_rot = attn.apply_rotary_position_embedding(
        q, k,
        attn.cos_sp[:, :4],
        attn.sin_sp[:, :4]
    )

    # Check output shapes
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

def test_swiglu_initialization():
    """Test that SwiGLU module is initialized correctly with paper's base_scale"""
    config = ModelConfiguration(
        num_layers=1,
        num_heads=2,
        d_model=32,
        vocab_size=50304,
        block_size=32,
        use_liger=False,
        rope_dtype=torch.float32
    )

    swiglu = SwiGLU(config)

    # Test initial scaling parameters
    expected_base_scale = 1 / (50304 ** 0.5)
    assert torch.allclose(
        swiglu.suv.data,
        torch.ones_like(swiglu.suv.data) * expected_base_scale
    )
    assert swiglu.suv_init_value == 1.0
    assert swiglu.suv_init_scaling == expected_base_scale

def test_swiglu_forward():
    """Test the complete SwiGLU forward pass"""
    config = ModelConfiguration(
        num_layers=1,
        num_heads=2,
        d_model=32,
        vocab_size=50304,
        block_size=32,
        use_liger=False,
        rope_dtype=torch.float32
    )

    swiglu = SwiGLU(config)

    # Test with different batch sizes and sequence lengths
    test_cases = [
        (1, 4),   # minimal case
        (2, 16),  # typical case
        (3, 32),  # full block size
    ]

    for batch_size, seq_len in test_cases:
        x = torch.randn(batch_size, seq_len, config.d_model)
        output = swiglu(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, config.d_model)

        # Check output is not None or NaN
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

def test_swiglu_gradients():
    """Test that gradients flow correctly through SwiGLU"""
    config = ModelConfiguration(
        num_layers=1,
        num_heads=2,
        d_model=32,
        vocab_size=50304,
        block_size=32,
        use_liger=False,
        rope_dtype=torch.float32
    )

    swiglu = SwiGLU(config)

    # Create input that requires grad
    x = torch.randn(2, 4, config.d_model, requires_grad=True)

    # Forward pass
    output = swiglu(x)

    # Create dummy loss and backward
    loss = output.sum()
    loss.backward()

    # Check that gradients exist and are not None
    assert x.grad is not None
    assert swiglu.suv.grad is not None
    assert not torch.isnan(swiglu.suv.grad).any()

def test_normalized_liger_swiglu_initialization():
    """Test that Normalized_LigerSwiGLUMLP is initialized correctly"""
    config = ModelConfiguration(
        num_layers=1,
        num_heads=2,
        d_model=32,
        vocab_size=50304,
        block_size=32,
        use_liger=True,
        rope_dtype=torch.float32
    )

    # Create a config object that matches LigerSwiGLUMLP expectations
    liger_config = type('Config', (), {
        'hidden_size': config.d_model,
        'intermediate_size': config.d_model * 4,
        'hidden_act': "silu",
        'n_embd': config.d_model,
    })()

    mlp = Normalized_LigerSwiGLUMLP(liger_config, config)

    # Test initial scaling parameters
    expected_base_scale = 1 / (50304 ** 0.5)
    assert torch.allclose(
        mlp.suv.data,
        torch.ones_like(mlp.suv.data) * expected_base_scale
    )
    assert mlp.suv_init_value == 1.0
    assert mlp.suv_init_scaling == expected_base_scale

def test_normalized_liger_swiglu_forward():
    """Test the complete Normalized_LigerSwiGLUMLP forward pass"""
    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = ModelConfiguration(
        num_layers=1,
        num_heads=2,
        d_model=32,
        vocab_size=50304,
        block_size=32,
        use_liger=True,
        rope_dtype=torch.float32
    )

    liger_config = type('Config', (), {
        'hidden_size': config.d_model,
        'intermediate_size': config.d_model * 4,
        'hidden_act': "silu",
        'n_embd': config.d_model,
    })()

    mlp = Normalized_LigerSwiGLUMLP(liger_config, config).cuda()  # Move model to GPU

    # Test with different batch sizes and sequence lengths
    test_cases = [
        (1, 4),   # minimal case
        (2, 16),  # typical case
        (3, 32),  # full block size
    ]

    for batch_size, seq_len in test_cases:
        x = torch.randn(batch_size, seq_len, config.d_model).cuda()  # Move input to GPU
        output = mlp(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, config.d_model)

        # Check output is not None or NaN
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

def test_normalized_liger_swiglu_gradients():
    """Test that gradients flow correctly through Normalized_LigerSwiGLUMLP"""
    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = ModelConfiguration(
        num_layers=1,
        num_heads=2,
        d_model=32,
        vocab_size=50304,
        block_size=32,
        use_liger=True,
        rope_dtype=torch.float32
    )

    liger_config = type('Config', (), {
        'hidden_size': config.d_model,
        'intermediate_size': config.d_model * 4,
        'hidden_act': "silu",
        'n_embd': config.d_model,
    })()

    mlp = Normalized_LigerSwiGLUMLP(liger_config, config).cuda()

    # Create input tensor and move to CUDA first, then set requires_grad
    x = torch.randn(2, 4, config.d_model).cuda()
    x.requires_grad_(True)  # This makes it a leaf tensor

    # Forward pass
    output = mlp(x)
    output.retain_grad()  # Retain grad for the output if needed

    # Create dummy loss and backward
    loss = output.sum()
    loss.backward()

    # Check that gradients exist and are not None
    assert x.grad is not None
    assert mlp.suv.grad is not None
    assert not torch.isnan(mlp.suv.grad).any()


def test_block_residual_connections():
    """Test that Block's residual connections maintain normalized form"""
    config = ModelConfiguration(
        num_layers=1,
        num_heads=2,
        d_model=32,
        vocab_size=50304,
        block_size=32,
        use_liger=False,
        rope_dtype=torch.float32
    )

    block = Block(config)

    # Input tensor
    x = torch.randn(2, 4, config.d_model)

    # Forward pass
    output = block(x)

    # Check that output is normalized
    output_norms = torch.norm(output, p=2, dim=-1)
    assert torch.allclose(output_norms, torch.ones_like(output_norms), atol=1e-6)

def test_model_sz_scaling():
    """Test that logits are properly scaled by sz parameter"""
    config = ModelConfiguration(
        num_layers=1,
        num_heads=2,
        d_model=32,
        vocab_size=50304,
        block_size=32,
        use_liger=False,
        rope_dtype=torch.float32
    )

    model = ModelBasis(config)
    x = torch.randint(0, config.vocab_size, (2, 4))

    # Get logits with and without scaling
    logits, _ = model(x)
    model.sz.data.fill_(1.0)  # Set scaling to 1.0 to get unscaled logits
    unscaled_logits, _ = model(x)

    # Verify the original scaling was applied
    expected_scale = 1 / (50304 ** 0.5)
    assert torch.allclose(logits, unscaled_logits * expected_scale)
