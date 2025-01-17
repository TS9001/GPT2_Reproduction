import torch
import pytest
from models.gpt_2_baseline_liger import GPT2Basic, GPT2Configuration, RMSNorm, Attention
from liger_kernel.transformers import LigerRMSNorm

def test_loss_equivalence():
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Setup
    config = GPT2Configuration(
        block_size=16,
        num_layers=2,
        num_heads=4,
        d_model=128,
        vocab_size=1000,
        use_liger=False
    )

    model = GPT2Basic(config).cuda()

    # Create sample input data on CUDA
    batch_size = 2
    seq_length = 8
    x_bsd = torch.randn(batch_size, seq_length, config.d_model, device='cuda')  # Create dummy hidden states
    y = torch.randint(0, config.vocab_size, (batch_size, seq_length), device='cuda')

    # Test standard loss computation directly
    standard_logits = model.lm_head(x_bsd)
    standard_loss = torch.nn.functional.cross_entropy(
        standard_logits.view(-1, config.vocab_size),
        y.view(-1)
    )

    # Test liger loss computation directly
    liger_loss = model.compute_liger_loss(x_bsd, y, return_logits=False)[1]

    # Compare losses
    torch.testing.assert_close(
        standard_loss,
        liger_loss,
        rtol=1e-4,
        atol=1e-4,
        msg="Liger loss and standard loss should be approximately equal"
    )

def test_rmsnorm_implementations():
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Setup dimensions
    batch_size = 2
    seq_length = 8
    d_model = 128
    eps = 1e-8

    # Create input tensor
    x = torch.randn(batch_size, seq_length, d_model, device='cuda')

    # Custom RMSNorm implementation
    custom_norm = RMSNorm(d_model, eps=eps).cuda()
    # Liger RMSNorm implementation
    liger_norm = LigerRMSNorm(d_model, eps=eps).cuda()

    # Make weights the same for fair comparison
    liger_norm.weight.data.copy_(custom_norm.weight_d.data)

    # Native PyTorch implementation
    def torch_rmsnorm(x, weight, eps):
        # Calculate RMSNorm: x * w * rsqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight

    # Get outputs from all implementations
    custom_output = custom_norm(x)
    liger_output = liger_norm(x)
    torch_output = torch_rmsnorm(x, custom_norm.weight_d, eps)

    # Compare custom vs torch
    torch.testing.assert_close(
        custom_output,
        torch_output,
        rtol=1e-4,
        atol=1e-4,
        msg="Custom RMSNorm and PyTorch implementation should be approximately equal"
    )

    # Compare custom vs liger
    torch.testing.assert_close(
        custom_output,
        liger_output,
        rtol=1e-4,
        atol=1e-4,
        msg="Custom RMSNorm and Liger RMSNorm should be approximately equal"
    )

    # Test with different input shapes
    shapes = [
        (1, 16, d_model),    # Different sequence length
        (4, 8, d_model),     # Different batch size
        (2, 8, d_model),     # Original shape
    ]

    for shape in shapes:
        x = torch.randn(*shape, device='cuda')
        custom_output = custom_norm(x)
        liger_output = liger_norm(x)
        torch_output = torch_rmsnorm(x, custom_norm.weight_d, eps)

        torch.testing.assert_close(
            custom_output,
            torch_output,
            rtol=1e-4,
            atol=1e-4,
            msg=f"Custom and PyTorch RMSNorm should match for shape {shape}"
        )

        torch.testing.assert_close(
            custom_output,
            liger_output,
            rtol=1e-4,
            atol=1e-4,
            msg=f"Custom and Liger RMSNorm should match for shape {shape}"
        )